package com.solquant.agent.config;

import com.solquant.agent.llm.FastApiChatModel;
import com.solquant.agent.service.EdgeMonitorAgent;
import com.solquant.agent.tools.AlertTool;
import com.solquant.agent.tools.SystemMetricsTool;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.service.AiServices;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Spring configuration that wires together:
 * <ol>
 *   <li>{@link FastApiChatModel} — custom LLM client pointing at the Python server</li>
 *   <li>{@link EdgeMonitorAgent} — LangChain4j AI Service with tool support</li>
 *   <li>Chat memory — sliding window of recent messages for context</li>
 * </ol>
 */
@Configuration
@EnableConfigurationProperties({LlmProperties.class, AlertProperties.class})
public class AgentConfig {

    private static final Logger log = LoggerFactory.getLogger(AgentConfig.class);

    /**
     * Register our custom {@link ChatLanguageModel} that calls FastAPI.
     */
    @Bean
    public ChatLanguageModel chatLanguageModel(LlmProperties llmProperties) {
        log.info("Creating FastApiChatModel → {}", llmProperties.generateUrl());
        return new FastApiChatModel(llmProperties);
    }

    /**
     * Build the LangChain4j agent proxy with tools and memory.
     * <p>
     * {@code AiServices} inspects the {@link EdgeMonitorAgent} interface,
     * discovers {@code @Tool}-annotated methods on the provided tool objects,
     * and generates an implementation that:
     * <ol>
     *   <li>Sends the user message + system prompt to the LLM</li>
     *   <li>Parses any tool-call requests from the LLM response</li>
     *   <li>Invokes the matching Java tool method</li>
     *   <li>Feeds the tool result back to the LLM</li>
     *   <li>Repeats until the LLM produces a final text response</li>
     * </ol>
     */
    @Bean
    public EdgeMonitorAgent edgeMonitorAgent(
            ChatLanguageModel chatLanguageModel,
            SystemMetricsTool systemMetricsTool,
            AlertTool alertTool
    ) {
        log.info("Building EdgeMonitorAgent with tools: [readSystemMetrics, writeAlert]");

        return AiServices.builder(EdgeMonitorAgent.class)
                .chatLanguageModel(chatLanguageModel)
                .tools(systemMetricsTool, alertTool)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
                .build();
    }
}
