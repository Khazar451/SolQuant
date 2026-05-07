package com.solquant.agent.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Alert subsystem configuration bound from {@code solquant.alerts.*}.
 */
@ConfigurationProperties(prefix = "solquant.alerts")
public record AlertProperties(
        String logFile
) {
}
