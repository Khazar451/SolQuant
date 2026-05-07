package com.solquant.agent.tools;

import com.solquant.agent.config.AlertProperties;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

/**
 * Tool for writing alerts to a local log file.
 * <p>
 * The AI agent invokes this tool when it detects anomalies in system metrics.
 * Each alert is appended to the configured log file with a timestamp,
 * severity level, and descriptive message.
 * <p>
 * Alert severities: INFO, WARNING, CRITICAL
 */
@Component
public class AlertTool {

    private static final Logger log = LoggerFactory.getLogger(AlertTool.class);
    private static final DateTimeFormatter TIMESTAMP_FMT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")
                    .withZone(ZoneId.systemDefault());

    private final Path alertLogPath;

    public AlertTool(AlertProperties alertProperties) {
        this.alertLogPath = Path.of(alertProperties.logFile());
        ensureLogDirectory();
        log.info("AlertTool initialised → {}", alertLogPath.toAbsolutePath());
    }

    @Tool("Write an alert to the system alert log file. Use this tool when you " +
          "detect any anomaly or concerning metric value. The severity must be " +
          "one of: INFO, WARNING, or CRITICAL. Provide a clear, actionable " +
          "message describing the issue and recommended action.")
    public String writeAlert(
            @P("Severity level: INFO, WARNING, or CRITICAL") String severity,
            @P("Descriptive alert message explaining the issue") String message
    ) {
        log.info("🚨 Tool invoked: writeAlert(severity={}, message={})", severity, message);

        // Validate severity
        String normalizedSeverity = severity.toUpperCase().trim();
        if (!normalizedSeverity.matches("INFO|WARNING|CRITICAL")) {
            normalizedSeverity = "WARNING"; // safe default
        }

        String timestamp = TIMESTAMP_FMT.format(Instant.now());
        String alertLine = String.format("[%s] [%-8s] %s%n",
                timestamp, normalizedSeverity, message);

        try {
            ensureLogDirectory();
            Files.writeString(
                    alertLogPath,
                    alertLine,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND
            );

            String confirmation = String.format(
                    "✓ Alert written successfully.\n" +
                    "  File:     %s\n" +
                    "  Severity: %s\n" +
                    "  Time:     %s\n" +
                    "  Message:  %s",
                    alertLogPath.toAbsolutePath(),
                    normalizedSeverity,
                    timestamp,
                    message
            );
            log.info("Alert persisted to {}", alertLogPath);
            return confirmation;

        } catch (IOException e) {
            String error = "Failed to write alert: " + e.getMessage();
            log.error(error, e);
            return "✗ " + error;
        }
    }

    private void ensureLogDirectory() {
        try {
            Path parent = alertLogPath.getParent();
            if (parent != null && !Files.exists(parent)) {
                Files.createDirectories(parent);
                log.info("Created alert log directory: {}", parent);
            }
        } catch (IOException e) {
            log.warn("Could not create log directory: {}", e.getMessage());
        }
    }
}
