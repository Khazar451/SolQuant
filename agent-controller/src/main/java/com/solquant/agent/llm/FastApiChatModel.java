package com.solquant.agent.llm;

import com.solquant.agent.config.LlmProperties;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.TokenUsage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.Duration;
import java.util.List;

/**
 * Custom {@link ChatLanguageModel} implementation that delegates inference
 * to our FastAPI Python server ({@code POST /generate}).
 * <p>
 * This bridges LangChain4j's chat abstraction with the Qwen2.5-0.5B model
 * running on the MX550 GPU via llama-cpp-python.
 * <p>
 * Message handling:
 * <ul>
 *   <li>{@link SystemMessage} → mapped to the {@code system_prompt} field</li>
 *   <li>{@link UserMessage} → concatenated into the {@code prompt} field</li>
 *   <li>{@link AiMessage} → included as assistant context in the prompt</li>
 * </ul>
 */
public class FastApiChatModel implements ChatLanguageModel {

    private static final Logger log = LoggerFactory.getLogger(FastApiChatModel.class);

    private final WebClient webClient;
    private final LlmProperties properties;

    public FastApiChatModel(LlmProperties properties) {
        this.properties = properties;
        this.webClient = WebClient.builder()
                .baseUrl(properties.baseUrl())
                .build();
        log.info("FastApiChatModel configured → {}", properties.generateUrl());
    }

    @Override
    public ChatResponse chat(ChatRequest chatRequest) {
        List<ChatMessage> messages = chatRequest.messages();

        // ── Extract system prompt and build conversation ────────────
        String systemPrompt = properties.systemPrompt();
        StringBuilder conversationBuilder = new StringBuilder();

        for (ChatMessage message : messages) {
            switch (message) {
                case SystemMessage sm -> systemPrompt = sm.text();
                case UserMessage um -> conversationBuilder
                        .append("User: ")
                        .append(um.singleText())
                        .append("\n");
                case AiMessage am -> conversationBuilder
                        .append("Assistant: ")
                        .append(am.text())
                        .append("\n");
                default -> log.warn("Unhandled message type: {}",
                        message.getClass().getSimpleName());
            }
        }

        String prompt = conversationBuilder.toString().trim();

        log.debug("Sending to FastAPI: prompt_len={}, system_len={}",
                prompt.length(), systemPrompt.length());

        // ── Call the FastAPI /generate endpoint ─────────────────────
        var request = new FastApiGenerateRequest(
                prompt,
                properties.maxTokens(),
                properties.temperature(),
                systemPrompt
        );

        FastApiGenerateResponse response;
        try {
            response = webClient.post()
                    .uri(properties.generatePath())
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(request)
                    .retrieve()
                    .bodyToMono(FastApiGenerateResponse.class)
                    .timeout(Duration.ofSeconds(properties.timeoutSeconds()))
                    .block();
        } catch (Exception e) {
            log.error("FastAPI call failed: {}", e.getMessage(), e);
            throw new RuntimeException("LLM inference failed: " + e.getMessage(), e);
        }

        if (response == null || response.text() == null) {
            throw new RuntimeException("Empty response from FastAPI /generate");
        }

        log.info("FastAPI response: {} tokens in {}s ({} tok/s)",
                response.tokensGenerated(),
                String.format("%.2f", response.generationTimeSec()),
                String.format("%.1f", response.tokensPerSecond()));

        // ── Map to LangChain4j response ────────────────────────────
        AiMessage aiMessage = AiMessage.from(response.text());
        TokenUsage tokenUsage = new TokenUsage(
                response.tokensPrompt(),
                response.tokensGenerated()
        );

        return ChatResponse.builder()
                .aiMessage(aiMessage)
                .tokenUsage(tokenUsage)
                .finishReason(FinishReason.STOP)
                .build();
    }
}
