FROM eclipse-temurin:17-jre

WORKDIR /app

COPY target/telemetry-producer-1.0-SNAPSHOT.jar app.jar

ENTRYPOINT ["java", "-jar", "app.jar"]