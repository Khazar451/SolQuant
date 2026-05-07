package com.solquant.agent.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Type-safe configuration properties bound from {@code solquant.llm.*}
 * in application.yml.
 */
@ConfigurationProperties(prefix = "solquant.llm")
public record LlmProperties(
        String baseUrl,
        String generatePath,
        int maxTokens,
        double temperature,
        int timeoutSeconds,
        String systemPrompt
) {
    /**
     * Resolve the full /generate endpoint URL.
     */
    public String generateUrl() {
        return baseUrl + generatePath;
    }
}
