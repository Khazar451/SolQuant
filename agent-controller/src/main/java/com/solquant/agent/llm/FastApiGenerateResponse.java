package com.solquant.agent.llm;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Maps the FastAPI {@code /generate} response body.
 *
 * <pre>
 * {
 *   "text": "...",
 *   "tokens_generated": 42,
 *   "tokens_prompt": 128,
 *   "generation_time_sec": 3.21,
 *   "tokens_per_second": 13.1,
 *   "vram": { ... }
 * }
 * </pre>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record FastApiGenerateResponse(
        String text,
        @JsonProperty("tokens_generated") int tokensGenerated,
        @JsonProperty("tokens_prompt") int tokensPrompt,
        @JsonProperty("generation_time_sec") double generationTimeSec,
        @JsonProperty("tokens_per_second") double tokensPerSecond
) {
}
