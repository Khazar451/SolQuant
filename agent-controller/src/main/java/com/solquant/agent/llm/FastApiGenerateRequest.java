package com.solquant.agent.llm;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Maps the FastAPI {@code /generate} request body.
 *
 * <pre>
 * {
 *   "prompt": "...",
 *   "max_tokens": 512,
 *   "temperature": 0.3,
 *   "system_prompt": "..."
 * }
 * </pre>
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record FastApiGenerateRequest(
        String prompt,
        @JsonProperty("max_tokens") int maxTokens,
        double temperature,
        @JsonProperty("system_prompt") String systemPrompt
) {
}
