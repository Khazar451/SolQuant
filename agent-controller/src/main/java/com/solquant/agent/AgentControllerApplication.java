package com.solquant.agent;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * SolQuant Agent Controller
 * <p>
 * Spring Boot application that uses LangChain4j to orchestrate an
 * autonomous AI agent backed by a custom Python LLM inference server.
 * The agent monitors edge system metrics and raises alerts autonomously.
 */
@SpringBootApplication
public class AgentControllerApplication {

    public static void main(String[] args) {
        SpringApplication.run(AgentControllerApplication.class, args);
    }
}
