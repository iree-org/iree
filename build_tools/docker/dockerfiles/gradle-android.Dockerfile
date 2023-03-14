# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for cross-compiling IREE's TFLite Java Bindings with Gradle and
# CMake.

FROM gcr.io/iree-oss/base@sha256:dcae1cb774c62680ffb9ed870a255181a428aacf5eb2387676146e055bc3b9e8

### Java ###
WORKDIR /install-jdk
# Download and install openjdk-11
ARG JDK_VERSION=11
RUN apt-get update && apt-get install -y openjdk-11-jdk

### Gradle ###
WORKDIR /install-gradle

ARG GRADLE_VERSION=7.1.1
ARG GRADLE_DIST=bin

# Download and install Gradle
RUN wget -q https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-${GRADLE_DIST}.zip && \
    unzip -q gradle*.zip -d /opt/ && \
    ln -s /opt/gradle-${GRADLE_VERSION}/bin/gradle /usr/bin/gradle \
    rm -rf /install-gradle

### Android ###
WORKDIR /install-android

# Download and install Android SDK
# Note: Uses the latest SDK version from https://developer.android.com/studio,
# however Gradle will automatically download any additional SDK/tooling versions
# as necessary.
ARG ANDROID_SDK_VERSION=7583922
ARG ANDROID_NDK_VERSION=21.4.7075529

ENV ANDROID_SDK_ROOT /opt/android-sdk
ENV ANDROID_HOME ${ANDROID_SDK_ROOT}
ENV ANDROID_NDK /opt/android-sdk/ndk/${ANDROID_NDK_VERSION}

RUN mkdir -p "${ANDROID_SDK_ROOT}/cmdline-tools" \
    && curl --silent --fail --show-error --location \
        "https://dl.google.com/android/repository/commandlinetools-linux-${ANDROID_SDK_VERSION}_latest.zip" \
        --output android_tools.zip \
    && unzip -q android_tools.zip -d "${ANDROID_SDK_ROOT}/cmdline-tools" \
    && mv "${ANDROID_SDK_ROOT}/cmdline-tools/cmdline-tools" "${ANDROID_SDK_ROOT}/cmdline-tools/tools" \
    # yes will give a non-zero exit code in non-interactive settings (broken pipe?)
    # with -o pipefail this leads to an error.
    && { yes || true; } | "${ANDROID_SDK_ROOT}/cmdline-tools/tools/bin/sdkmanager" --licenses \
    && /opt/android-sdk/cmdline-tools/tools/bin/sdkmanager --install "ndk;${ANDROID_NDK_VERSION}" \
    && rm -rf /install-android

WORKDIR /
