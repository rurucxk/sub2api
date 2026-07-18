package service

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	grokConversationIDHeader        = "X-Grok-Conv-Id"
	grokFreeCacheNativeToolsJSON    = `[{"type":"web_search"},{"type":"x_search"}]`
	grokFreeCacheDisabledToolChoice = "none"
	grokFreeRolling24hTokenLimit    = int64(2_000_000)

	// Plan B + A: keep main-dialogue prompt_cache_key and tools snapshot across
	// Grok Build idle recap so upstream prompt cache is not broken by stripped tools.
	grokMainCacheIdentityTTL = 2 * time.Hour
	// Cap stored tools JSON to avoid unbounded memory if a client floods tool schemas.
	grokMainToolsSnapshotMaxBytes = 1 << 20
)

type grokMainCacheIdentityEntry struct {
	identity  string
	toolsRaw  string // plan A: last main-dialogue tools array JSON
	expiresAt time.Time
}

var grokMainCacheIdentityMu sync.Mutex
var grokMainCacheIdentityByAPIKey = map[int64]grokMainCacheIdentityEntry{}

// resolveGrokCacheIdentity derives one stable, tenant-isolated routing identity
// for xAI's server-side prompt cache. The returned value is safe to expose to
// the upstream: it never contains the client's raw session identifier.
//
// A valid downstream API key is required. This intentionally fails closed on
// internal probes and incomplete request contexts instead of creating a cache
// identity that could be shared by unrelated tenants.
//
// Idle-recap-like Grok Build requests often strip client function tools, which
// changes the content-derived seed and would mint a new UUID. When possible we
// reuse the last main-dialogue identity for the same API key (plan B).
func resolveGrokCacheIdentity(c *gin.Context, body []byte, explicitKey, upstreamModel string) string {
	apiKeyID := getAPIKeyIDFromContext(c)
	if apiKeyID <= 0 {
		return ""
	}
	// /responses/compact rejects tool_choice and does not represent a normal
	// conversation turn. Keep both cache identity and Free-tier routing
	// augmentation out of this path.
	if isOpenAIResponsesCompactPath(c) {
		return ""
	}

	model := strings.ToLower(strings.TrimSpace(upstreamModel))
	if model == "" {
		return ""
	}

	seed := explicitGrokCacheSeed(c, body, explicitKey)
	if seed == "" {
		seed = deriveOpenAIStablePrefixSessionSeed(body)
		if seed == "" {
			// A model alone is too broad for cache routing. Preserve the
			// existing first-user-derived identity when no reusable prefix is
			// available so unrelated prompts do not share one tenant-wide key.
			seed = deriveOpenAIAnchoredContentSessionSeed(body)
		}
	}
	if seed == "" {
		return ""
	}

	// generateSessionUUID hashes the whole seed before formatting it as a UUID.
	// Include a versioned namespace so this identity cannot collide with other
	// upstream session identifiers derived by sub2api.
	isolatedSeed := fmt.Sprintf("grok-prompt-cache:v1:%d:%s:%s", apiKeyID, model, seed)
	identity := generateSessionUUID(isolatedSeed)
	return applyGrokIdleStickyCacheIdentity(apiKeyID, body, identity)
}

// applyGrokIdleStickyCacheIdentity implements plan B:
//   - main dialogue (has multiple function tools): remember identity
//   - idle-recap-like (stripped tools / idle marker): reuse remembered identity
func applyGrokIdleStickyCacheIdentity(apiKeyID int64, body []byte, identity string) string {
	if apiKeyID <= 0 {
		return identity
	}
	if isGrokIdleRecapLikeRequest(body) {
		if prev := loadGrokMainCacheIdentity(apiKeyID); prev != "" {
			return prev
		}
		return identity
	}
	if identity != "" && isGrokMainDialogueCacheRequest(body) {
		storeGrokMainCacheIdentity(apiKeyID, identity)
	}
	return identity
}

func storeGrokMainCacheIdentity(apiKeyID int64, identity string) {
	identity = strings.TrimSpace(identity)
	if apiKeyID <= 0 || identity == "" {
		return
	}
	grokMainCacheIdentityMu.Lock()
	defer grokMainCacheIdentityMu.Unlock()
	pruneGrokMainCacheIdentityLocked(time.Now())
	entry := grokMainCacheIdentityByAPIKey[apiKeyID]
	entry.identity = identity
	entry.expiresAt = time.Now().Add(grokMainCacheIdentityTTL)
	grokMainCacheIdentityByAPIKey[apiKeyID] = entry
}

func loadGrokMainCacheIdentity(apiKeyID int64) string {
	if apiKeyID <= 0 {
		return ""
	}
	grokMainCacheIdentityMu.Lock()
	defer grokMainCacheIdentityMu.Unlock()
	entry, ok := grokMainCacheIdentityByAPIKey[apiKeyID]
	if !ok {
		return ""
	}
	if time.Now().After(entry.expiresAt) {
		delete(grokMainCacheIdentityByAPIKey, apiKeyID)
		return ""
	}
	return entry.identity
}

func storeGrokMainToolsSnapshot(apiKeyID int64, toolsRaw string) {
	toolsRaw = strings.TrimSpace(toolsRaw)
	if apiKeyID <= 0 || toolsRaw == "" {
		return
	}
	if len(toolsRaw) > grokMainToolsSnapshotMaxBytes {
		return
	}
	// Must be a JSON array; reject garbage so idle rewrite cannot corrupt upstream.
	if !gjson.Valid(toolsRaw) || !gjson.Parse(toolsRaw).IsArray() {
		return
	}
	grokMainCacheIdentityMu.Lock()
	defer grokMainCacheIdentityMu.Unlock()
	pruneGrokMainCacheIdentityLocked(time.Now())
	entry := grokMainCacheIdentityByAPIKey[apiKeyID]
	entry.toolsRaw = toolsRaw
	entry.expiresAt = time.Now().Add(grokMainCacheIdentityTTL)
	grokMainCacheIdentityByAPIKey[apiKeyID] = entry
}

func loadGrokMainToolsSnapshot(apiKeyID int64) string {
	if apiKeyID <= 0 {
		return ""
	}
	grokMainCacheIdentityMu.Lock()
	defer grokMainCacheIdentityMu.Unlock()
	entry, ok := grokMainCacheIdentityByAPIKey[apiKeyID]
	if !ok {
		return ""
	}
	if time.Now().After(entry.expiresAt) {
		delete(grokMainCacheIdentityByAPIKey, apiKeyID)
		return ""
	}
	return entry.toolsRaw
}

func pruneGrokMainCacheIdentityLocked(now time.Time) {
	for k, v := range grokMainCacheIdentityByAPIKey {
		if now.After(v.expiresAt) {
			delete(grokMainCacheIdentityByAPIKey, k)
		}
	}
}

// resetGrokMainCacheIdentityStoreForTest clears sticky identities (unit tests).
func resetGrokMainCacheIdentityStoreForTest() {
	grokMainCacheIdentityMu.Lock()
	defer grokMainCacheIdentityMu.Unlock()
	grokMainCacheIdentityByAPIKey = map[int64]grokMainCacheIdentityEntry{}
}

// applyGrokIdleStickyToolsPlanA implements plan A:
//   - main dialogue: remember the final upstream tools array for this API key
//   - idle recap: rewrite stripped 2-tool payloads back to the remembered tools
//     so xAI prompt-cache prefix still matches; remove the idle-only
//     tool_choice=none field so the following normal turn keeps the same shape.
//
// intentSourceBody is the pre-patch client body (idle markers / tool intent).
// patchedBody is the body about to be sent upstream (may already include free-tier
// native tools append).
func applyGrokIdleStickyToolsPlanA(apiKeyID int64, patchedBody, intentSourceBody []byte) ([]byte, error) {
	if apiKeyID <= 0 || len(patchedBody) == 0 {
		return patchedBody, nil
	}
	detectBody := intentSourceBody
	if len(detectBody) == 0 {
		detectBody = patchedBody
	}

	if isGrokIdleRecapLikeRequest(detectBody) {
		snap := loadGrokMainToolsSnapshot(apiKeyID)
		if snap == "" {
			return patchedBody, nil
		}
		out, err := sjson.SetRawBytes(patchedBody, "tools", []byte(snap))
		if err != nil {
			return nil, err
		}
		if strings.EqualFold(strings.TrimSpace(gjson.GetBytes(out, "tool_choice").String()), grokFreeCacheDisabledToolChoice) {
			return sjson.DeleteBytes(out, "tool_choice")
		}
		return out, nil
	}

	// Remember tools from main multi-function dialogue after free-tier merge.
	if isGrokMainDialogueCacheRequest(detectBody) || isGrokMainDialogueCacheRequest(patchedBody) {
		tools := gjson.GetBytes(patchedBody, "tools")
		if tools.Exists() && tools.IsArray() && len(tools.Array()) >= 3 {
			storeGrokMainToolsSnapshot(apiKeyID, tools.Raw)
		}
	}
	return patchedBody, nil
}

// isGrokIdleRecapLikeRequest detects Grok Build idle recap / stripped-tool turns
// that should reuse the previous main-dialogue cache routing identity.
func isGrokIdleRecapLikeRequest(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	raw := string(body)
	if strings.Contains(raw, "returning from idle") ||
		strings.Contains(raw, "Write ONE sentence recap") ||
		strings.Contains(raw, "Recap —") ||
		strings.Contains(raw, "Recap -") {
		return true
	}

	tools := gjson.GetBytes(body, "tools")
	if !tools.Exists() || !tools.IsArray() || len(tools.Array()) == 0 {
		// Tool-less alone is not enough (could be a probe); require idle markers.
		return false
	}

	functionTools := 0
	onlyNativeSearch := true
	for _, tool := range tools.Array() {
		typ := strings.TrimSpace(tool.Get("type").String())
		switch typ {
		case "function":
			functionTools++
			onlyNativeSearch = false
		case "web_search", "x_search":
			// ok
		default:
			onlyNativeSearch = false
		}
	}
	if functionTools > 0 {
		return false
	}
	// Native-search-only with tool_choice=none is how free-tier injects idle-like
	// requests after stripping client tools; still require an idle-ish signal when
	// markers above did not match, to avoid false positives on pure tool-less tests.
	_ = onlyNativeSearch
	return false
}

// isGrokMainDialogueCacheRequest reports whether the request carries a normal
// multi-function-tool Grok Build / Claude Code dialogue shape worth remembering.
func isGrokMainDialogueCacheRequest(body []byte) bool {
	tools := gjson.GetBytes(body, "tools")
	if !tools.Exists() || !tools.IsArray() {
		return false
	}
	functionTools := 0
	for _, tool := range tools.Array() {
		if strings.TrimSpace(tool.Get("type").String()) == "function" {
			functionTools++
		}
	}
	return functionTools >= 3
}

func explicitGrokCacheSeed(c *gin.Context, body []byte, explicitKey string) string {
	seed := ""
	if c != nil {
		seed = strings.TrimSpace(c.GetHeader("session_id"))
		if seed == "" {
			seed = strings.TrimSpace(c.GetHeader("conversation_id"))
		}
		if seed == "" {
			seed = strings.TrimSpace(c.GetHeader(grokConversationIDHeader))
		}
	}
	if seed == "" && len(body) > 0 {
		seed = strings.TrimSpace(gjson.GetBytes(body, "prompt_cache_key").String())
	}
	if seed == "" {
		seed = strings.TrimSpace(explicitKey)
	}
	return seed
}

func isGrokRequestContext(c *gin.Context) bool {
	if c == nil {
		return false
	}
	v, exists := c.Get("api_key")
	if !exists {
		return false
	}
	apiKey, ok := v.(*APIKey)
	return ok && apiKey != nil && apiKey.Group != nil && apiKey.Group.Platform == PlatformGrok
}

// applyGrokResponsesCacheIdentity writes the cache routing identity into an
// xAI Responses request. Existing client values are deliberately replaced by
// the tenant-isolated value to prevent collisions on shared OAuth accounts.
//
// Free OAuth requests without native search tools are routed by xAI to the
// non-cacheable build-free model. For otherwise tool-free requests, add the
// native tools with tool_choice=none: this selects the cache-capable tier
// without allowing an actual search. Explicit client function tools are handled by
// applyGrokFreeMessagesFunctionToolCacheRoute (Messages bridge and native Responses).
func applyGrokResponsesCacheIdentity(body, intentSourceBody []byte, identity string, injectFreeTierTools bool) ([]byte, error) {
	identity = strings.TrimSpace(identity)
	if identity == "" {
		if gjson.GetBytes(body, "prompt_cache_key").Exists() {
			return sjson.DeleteBytes(body, "prompt_cache_key")
		}
		return body, nil
	}
	out, err := sjson.SetBytes(body, "prompt_cache_key", identity)
	if err != nil {
		return nil, err
	}
	if !injectFreeTierTools {
		return out, nil
	}
	// Inspect the pre-sanitization source. patchGrokResponsesBody may remove an
	// unsupported client tool and its tool_choice; that must not turn an
	// explicit client tool intent into an eligible native-tool request.
	if gjson.GetBytes(intentSourceBody, "tools").Exists() || gjson.GetBytes(intentSourceBody, "tool_choice").Exists() {
		return out, nil
	}
	out, err = sjson.SetRawBytes(out, "tools", []byte(grokFreeCacheNativeToolsJSON))
	if err != nil {
		return nil, err
	}
	return sjson.SetBytes(out, "tool_choice", grokFreeCacheDisabledToolChoice)
}

// applyGrokFreeMessagesFunctionToolCacheRoute enables xAI's cache-capable
// mixed-tools route only for the Anthropic Messages bridge and only when the
// selected account is known to be Free. Native tools become eligible under
// auto selection, so callers must not apply this policy to paid accounts or
// other ingress protocols implicitly.
func applyGrokFreeMessagesFunctionToolCacheRoute(body, intentSourceBody []byte, account *Account, cacheIdentity string) ([]byte, error) {
	if strings.TrimSpace(cacheIdentity) == "" || !isKnownGrokFreeAccount(account) {
		return body, nil
	}
	intentTools := gjson.GetBytes(intentSourceBody, "tools")
	intentToolChoice := gjson.GetBytes(intentSourceBody, "tool_choice")
	if !isGrokFreeCacheFunctionToolIntent(intentTools, intentToolChoice) {
		return body, nil
	}
	return appendMissingGrokFreeCacheNativeTools(body)
}

func isKnownGrokFreeAccount(account *Account) bool {
	if account == nil || !account.IsGrokOAuth() {
		return false
	}
	freeSignal := false
	paidSignal := false
	inferredFreeSignal := false
	if billing, err := grokBillingSnapshotFromExtra(account.Extra); err == nil && billing != nil {
		if tier := strings.TrimSpace(billing.Plan); tier != "" {
			if isGrokFreeSubscriptionTier(tier) {
				freeSignal = true
			} else if !isGrokUnknownSubscriptionTier(tier) {
				paidSignal = true
			}
		}
		if billing.UsagePercent != nil || billing.UsedPercent != nil ||
			(billing.MonthlyLimitCents != nil && *billing.MonthlyLimitCents > 0) {
			paidSignal = true
		}
		// xAI deliberately reports an empty plan for Free accounts; only paid
		// subscriptions receive a SuperGrok plan/monthly limit. A successful
		// monthly billing observation with no paid signal is therefore positive
		// Free evidence, not an unknown tier. Keep partial probes fail-closed.
		if strings.TrimSpace(billing.MonthlyUpdatedAt) != "" ||
			(billing.StatusCode >= http.StatusOK && billing.StatusCode < http.StatusMultipleChoices &&
				!billing.Partial && len(billing.FailedWindows) == 0) {
			inferredFreeSignal = true
		}
	}
	if snapshot, err := grokQuotaSnapshotFromExtra(account.Extra); err == nil && snapshot != nil {
		if tier := strings.TrimSpace(snapshot.SubscriptionTier); tier != "" {
			if isGrokFreeSubscriptionTier(tier) {
				freeSignal = true
			} else if !isGrokUnknownSubscriptionTier(tier) {
				paidSignal = true
			}
		}
		if snapshot.Tokens != nil && snapshot.Tokens.Limit != nil &&
			*snapshot.Tokens.Limit == grokFreeRolling24hTokenLimit {
			inferredFreeSignal = true
		}
	}
	if tier := strings.TrimSpace(account.GetCredential("subscription_tier")); tier != "" {
		if isGrokFreeSubscriptionTier(tier) {
			freeSignal = true
		} else if !isGrokUnknownSubscriptionTier(tier) {
			paidSignal = true
		}
	}
	// Explicit paid evidence always wins over an inferred Free signal. This
	// protects upgraded/stale accounts whose previous quota snapshot still
	// carries the historical 2M Free token limit.
	return !paidSignal && (freeSignal || inferredFreeSignal)
}

func isGrokFreeSubscriptionTier(tier string) bool {
	switch strings.ToLower(strings.TrimSpace(tier)) {
	case "free", "grok-free", "grok_free", "free-tier", "free_tier", "basic", "grok-basic", "grok_basic":
		return true
	default:
		return false
	}
}

func isGrokUnknownSubscriptionTier(tier string) bool {
	switch strings.ToLower(strings.TrimSpace(tier)) {
	case "", "unknown", "n/a", "none":
		return true
	default:
		return false
	}
}

func isGrokFreeCacheFunctionToolIntent(tools, toolChoice gjson.Result) bool {
	if !tools.IsArray() {
		return false
	}
	items := tools.Array()
	if len(items) == 0 {
		return false
	}
	for _, tool := range items {
		if !tool.IsObject() || strings.TrimSpace(tool.Get("type").String()) != "function" {
			return false
		}
		// Responses function declarations keep name at the top level. Reject
		// Chat Completions' nested function shape and incomplete declarations.
		if strings.TrimSpace(tool.Get("name").String()) == "" || tool.Get("function").Exists() {
			return false
		}
	}
	if !toolChoice.Exists() {
		return true
	}
	return toolChoice.Type == gjson.String && strings.TrimSpace(toolChoice.String()) == "auto"
}

func appendMissingGrokFreeCacheNativeTools(body []byte) ([]byte, error) {
	tools := gjson.GetBytes(body, "tools")
	if !tools.Exists() || !tools.IsArray() {
		return body, nil
	}

	items := tools.Array()
	if len(items) == 0 {
		return body, nil
	}
	merged := make([]json.RawMessage, 0, len(items)+2)
	present := make(map[string]bool, 2)
	hasFunction := false
	for _, tool := range items {
		toolType := strings.TrimSpace(tool.Get("type").String())
		switch toolType {
		case "function":
			name := strings.TrimSpace(tool.Get("name").String())
			if !tool.IsObject() || name == "" || tool.Get("function").Exists() {
				return body, nil
			}
			// Grok Build may declare search as function tools. Convert to native
			// entries so Free OAuth stays cache-capable without duplicate names.
			if name == "web_search" || name == "x_search" {
				if present[name] {
					continue
				}
				raw, err := json.Marshal(map[string]string{"type": name})
				if err != nil {
					return nil, err
				}
				merged = append(merged, raw)
				present[name] = true
				continue
			}
			hasFunction = true
			merged = append(merged, json.RawMessage(tool.Raw))
		case "web_search", "x_search":
			if present[toolType] {
				continue
			}
			merged = append(merged, json.RawMessage(tool.Raw))
			present[toolType] = true
		default:
			return body, nil
		}
	}
	if !hasFunction {
		return body, nil
	}
	// Only complement missing native search tools when the request already contains
	// at least one search tool (native or function-form). Pure client function tools
	// (e.g. view_image) must not trigger injection to avoid biasing model tool
	// selection (#4486).
	if !present["web_search"] && !present["x_search"] {
		return body, nil
	}
	for _, toolType := range []string{"web_search", "x_search"} {
		if present[toolType] {
			continue
		}
		raw, err := json.Marshal(map[string]string{"type": toolType})
		if err != nil {
			return nil, err
		}
		merged = append(merged, raw)
	}
	encoded, err := json.Marshal(merged)
	if err != nil {
		return nil, err
	}
	return sjson.SetRawBytes(body, "tools", encoded)
}

// applyGrokCacheHeaders applies the documented Chat Completions conversation
// routing header. The request is built from a fresh header map, so client
// supplied x-grok headers cannot override this server-derived value.
func applyGrokCacheHeaders(headers http.Header, identity string) {
	if headers == nil {
		return
	}
	identity = strings.TrimSpace(identity)
	if identity == "" {
		headers.Del(grokConversationIDHeader)
		return
	}
	headers.Set(grokConversationIDHeader, identity)
}

// stripGrokChatPromptCacheKey removes the Responses-only body field after it
// has been used as an identity seed. Chat Completions routes cache by header.
func stripGrokChatPromptCacheKey(body []byte) ([]byte, error) {
	if !gjson.GetBytes(body, "prompt_cache_key").Exists() {
		return body, nil
	}
	return sjson.DeleteBytes(body, "prompt_cache_key")
}
