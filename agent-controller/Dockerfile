# ═══════════════════════════════════════════════════════════════════
# SolQuant Agent Controller — Dockerfile
# ═══════════════════════════════════════════════════════════════════
# Multi-stage build for the Spring Boot + LangChain4j orchestrator.
#
# Stage 1: Maven build with dependency caching
# Stage 2: Slim JRE runtime
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: Build with Maven ──────────────────────────────────
FROM eclipse-temurin:21-jdk AS builder

WORKDIR /build

# Copy Maven wrapper and POM first for dependency caching
COPY pom.xml ./
COPY .mvn/ .mvn/
COPY mvnw ./
RUN chmod +x mvnw

# Download dependencies (cached unless pom.xml changes)
RUN ./mvnw dependency:go-offline -B

# Copy source and build
COPY src/ src/
RUN ./mvnw package -DskipTests -B && \
    mv target/*.jar app.jar


# ── Stage 2: Slim runtime ─────────────────────────────────────
FROM eclipse-temurin:21-jre

# Create non-root user
RUN groupadd -r solquant && useradd -r -g solquant -m solquant

WORKDIR /app

# Copy the fat JAR from builder
COPY --from=builder /build/app.jar ./app.jar

# Create logs directory
RUN mkdir -p /app/logs && chown -R solquant:solquant /app

USER solquant

# Spring Boot default port (configured to 8081 in application.yml)
EXPOSE 8081

# Health check against the agent health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8081/api/agent/health || exit 1

# JVM tuning for containerized environment
# -XX:MaxRAMPercentage: use up to 75% of container memory for heap
# -XX:+UseG1GC: low-latency GC suitable for API workloads
ENTRYPOINT ["java", \
    "-XX:MaxRAMPercentage=75.0", \
    "-XX:+UseG1GC", \
    "-XX:+UseContainerSupport", \
    "-jar", "app.jar"]
