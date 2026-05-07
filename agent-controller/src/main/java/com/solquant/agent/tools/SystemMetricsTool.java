package com.solquant.agent.tools;

import dev.langchain4j.agent.tool.Tool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.lang.management.ManagementFactory;
import java.lang.management.OperatingSystemMXBean;
import java.time.Instant;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Tool for reading system metrics — CPU load, RAM usage, disk, and temperature.
 * <p>
 * Uses a combination of real JVM metrics (where available) and simulated
 * sensor data for edge-device values (CPU temperature, disk I/O) that
 * aren't accessible through standard Java APIs.
 * <p>
 * In production, these would be sourced from {@code /proc}, SNMP, or
 * a native agent (e.g., Prometheus node_exporter).
 */
@Component
public class SystemMetricsTool {

    private static final Logger log = LoggerFactory.getLogger(SystemMetricsTool.class);
    private static final Random RNG = new Random();

    // Simulated metric history for trend detection
    private final Map<String, Double> lastValues = new ConcurrentHashMap<>();

    @Tool("Read current CPU and RAM metrics from the edge system. " +
          "Returns a detailed report including CPU load percentage, " +
          "RAM usage in MB, disk usage percentage, CPU temperature in Celsius, " +
          "and system uptime. Use this tool whenever you need to assess " +
          "the current health of the system.")
    public String readSystemMetrics() {
        log.info("📊 Tool invoked: readSystemMetrics()");

        // ── Real metrics from JVM ──────────────────────────────────
        OperatingSystemMXBean os = ManagementFactory.getOperatingSystemMXBean();
        Runtime runtime = Runtime.getRuntime();

        double cpuLoad = os.getSystemLoadAverage();
        if (cpuLoad < 0) {
            // Fallback: simulate if OS doesn't report load average
            cpuLoad = 15.0 + RNG.nextDouble() * 70.0;
        }
        // Normalize to percentage (load average can exceed 100% on multi-core)
        double cpuPercent = Math.min(cpuLoad * 100.0 / os.getAvailableProcessors(), 100.0);

        long totalMemMb = runtime.totalMemory() / (1024 * 1024);
        long freeMemMb = runtime.freeMemory() / (1024 * 1024);
        long usedMemMb = totalMemMb - freeMemMb;
        long maxMemMb = runtime.maxMemory() / (1024 * 1024);

        // ── Simulated edge-device metrics ──────────────────────────
        double cpuTempC = simulateMetric("cpu_temp", 42.0, 89.0);
        double diskUsagePct = simulateMetric("disk_usage", 30.0, 95.0);
        double networkMbps = simulateMetric("network_throughput", 0.5, 120.0);

        // ── Uptime ─────────────────────────────────────────────────
        long uptimeMs = ManagementFactory.getRuntimeMXBean().getUptime();
        long uptimeMin = uptimeMs / 60000;

        // ── Format the report ──────────────────────────────────────
        String report = String.format("""
                === SYSTEM METRICS REPORT ===
                Timestamp:       %s
                CPU Load:        %.1f%%
                CPU Temperature: %.1f°C %s
                RAM Used:        %d MB / %d MB (max %d MB)
                Disk Usage:      %.1f%% %s
                Network I/O:     %.1f Mbps
                Uptime:          %d minutes
                Processors:      %d cores
                =============================""",
                Instant.now(),
                cpuPercent,
                cpuTempC, cpuTempC > 80 ? "⚠ HIGH" : cpuTempC > 70 ? "⚡ ELEVATED" : "✓ NORMAL",
                usedMemMb, totalMemMb, maxMemMb,
                diskUsagePct, diskUsagePct > 85 ? "⚠ HIGH" : "✓ OK",
                networkMbps,
                uptimeMin,
                os.getAvailableProcessors()
        );

        log.info("Metrics collected — CPU: {}%, Temp: {}°C, RAM: {}MB/{}MB, Disk: {}%",
                String.format("%.1f", cpuPercent),
                String.format("%.1f", cpuTempC),
                usedMemMb, totalMemMb,
                String.format("%.1f", diskUsagePct));

        return report;
    }

    /**
     * Simulate a metric value with slight drift from last reading,
     * producing more realistic sensor data than pure random.
     */
    private double simulateMetric(String name, double min, double max) {
        double lastVal = lastValues.getOrDefault(name, (min + max) / 2);
        double drift = (RNG.nextGaussian() * (max - min) * 0.05);
        double newVal = Math.max(min, Math.min(max, lastVal + drift));
        lastValues.put(name, newVal);
        return Math.round(newVal * 10.0) / 10.0;
    }
}
