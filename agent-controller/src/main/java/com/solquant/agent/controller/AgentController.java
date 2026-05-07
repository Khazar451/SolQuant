package com.solquant.agent.controller;

import com.solquant.agent.service.EdgeMonitorAgent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;

/**
 * REST controller that exposes the AI agent's capabilities via HTTP.
 * <p>
 * Endpoints:
 * <ul>
 *   <li>{@code POST /api/agent/chat} — Send a message to the agent and
 *       receive its response (may include autonomous tool invocations)</li>
 *   <li>{@code GET /api/agent/health} — Check if the agent is ready</li>
 * </ul>
 *
 * The agent autonomously decides which tools to call based on the user's
 * query. For example, asking "How is the system doing?" will trigger:
 * <ol>
 *   <li>readSystemMetrics() → gets CPU, RAM, temp data</li>
 *   <li>writeAlert() → if any anomaly is detected</li>
 *   <li>Final summary → synthesised from real tool outputs</li>
 * </ol>
 */
@RestController
@RequestMapping("/api/agent")
public class AgentController {

    private static final Logger log = LoggerFactory.getLogger(AgentController.class);

    private final EdgeMonitorAgent agent;

    public AgentController(EdgeMonitorAgent agent) {
        this.agent = agent;
    }

    /**
     * Send a query to the edge monitoring agent.
     * <p>
     * The agent will autonomously decide whether to:
     * - Read system metrics
     * - Write alerts
     * - Simply respond with knowledge
     *
     * @param request The user's query
     * @return The agent's response with metadata
     */
    @PostMapping("/chat")
    public ResponseEntity<AgentResponse> chat(@RequestBody ChatRequest request) {
        log.info("─── Agent request ───────────────────────────────");
        log.info("Query: {}", request.query());

        Instant start = Instant.now();

        try {
            String response = agent.chat(request.query());
            Duration elapsed = Duration.between(start, Instant.now());

            log.info("Agent responded in {}ms", elapsed.toMillis());
            log.info("─────────────────────────────────────────────────");

            return ResponseEntity.ok(new AgentResponse(
                    response,
                    elapsed.toMillis(),
                    true,
                    null
            ));

        } catch (Exception e) {
            Duration elapsed = Duration.between(start, Instant.now());
            log.error("Agent error after {}ms: {}", elapsed.toMillis(), e.getMessage(), e);

            return ResponseEntity.internalServerError().body(new AgentResponse(
                    null,
                    elapsed.toMillis(),
                    false,
                    e.getMessage()
            ));
        }
    }

    /**
     * Health check for the agent controller.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        return ResponseEntity.ok(Map.of(
                "status", "ready",
                "service", "solquant-agent-controller",
                "agent", "EdgeMonitorAgent",
                "tools", new String[]{"readSystemMetrics", "writeAlert"}
        ));
    }

    // ── Request / Response records ─────────────────────────────────────

    public record ChatRequest(String query) {
    }

    public record AgentResponse(
            String response,
            long elapsedMs,
            boolean success,
            String error
    ) {
    }
}
