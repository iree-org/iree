# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE's TFLite Java Bindings with Gradle and
# CMake.

ARG JDK_VERSION=11

ARG GRADLE_VERSION=7.1.1
ARG GRADLE_DIST=bin

ARG ANDROID_SDK_VERSION=7583922
ARG ANDROID_NDK_VERSION=21.4.7075529

FROM gcr.io/iree-oss/base@sha256:8bd3dfa4e16c232802842c36f1139730e183f4263b709e76fec04df30de930db AS install-deps
ARG GRADLE_VERSION
ARG GRADLE_DIST
ARG ANDROID_SDK_VERSION

# Download and install Gradle
RUN cd /opt && \
    wget -q https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-${GRADLE_DIST}.zip && \
    unzip -q gradle*.zip && \
    rm gradle*.zip && \
    ln -s /opt/gradle-${GRADLE_VERSION}/bin/gradle /usr/bin/gradle

# Download and install Android SDK
# Note: Uses the latest SDK version from https://developer.android.com/studio,
# however Gradle will automatically download any additional SDK/tooling versions
# as necessary.
ENV ANDROID_SDK_ROOT /opt/android-sdk
RUN mkdir -p ${ANDROID_SDK_ROOT}/cmdline-tools && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-${ANDROID_SDK_VERSION}_latest.zip && \
    unzip -q *tools*linux*.zip -d ${ANDROID_SDK_ROOT}/cmdline-tools && \
    mv ${ANDROID_SDK_ROOT}/cmdline-tools/cmdline-tools ${ANDROID_SDK_ROOT}/cmdline-tools/tools && \
    rm *tools*linux*.zip

FROM gcr.io/iree-oss/base@sha256:8bd3dfa4e16c232802842c36f1139730e183f4263b709e76fec04df30de930db as final
ARG ANDROID_SDK_VERSION
ARG ANDROID_NDK_VERSION
ARG GRADLE_VERSION
ENV ANDROID_SDK_ROOT /opt/android-sdk

# Download and install openjdk-11
ARG JDK_VERSION
RUN apt-get update && apt-get install -y openjdk-11-jdk

# Copy /opt/ (Gradle + Android SDK) thenlink Gradle bin
COPY --from=install-deps /opt/ /opt/
RUN  ln -s /opt/gradle-${GRADLE_VERSION}/bin/gradle /usr/bin/gradle

# Accept the license agreements of the Android SDK components
RUN yes | ${ANDROID_SDK_ROOT}/cmdline-tools/tools/bin/sdkmanager --licenses

# Install the Android NDK
RUN /opt/android-sdk/cmdline-tools/tools/bin/sdkmanager --install "ndk;$ANDROID_NDK_VERSION"

# Define environment variables for the NDK/SDK
ENV ANDROID_HOME ${ANDROID_SDK_ROOT}
ENV ANDROID_NDK /opt/android-sdk/ndk/${ANDROID_NDK_VERSION}
