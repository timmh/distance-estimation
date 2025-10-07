#!/usr/bin/env bash

set -exv

required_vars=(
    APPLE_CODESIGNING_CERTIFICATE
    APPLE_CODESIGNING_PASSWORD
    APPLE_CODESIGNING_ID
    APPLE_NOTARIZATION_USER_ID
    APPLE_NOTARIZATION_TEAM_ID
    APPLE_NOTARIZATION_PASSWORD
)

for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "Environment variable '$var' is required but not set." >&2
        exit 1
    fi
done

echo "$APPLE_CODESIGNING_CERTIFICATE" | base64 --decode > macos_codesigning.p12
security create-keychain -p "$APPLE_CODESIGNING_PASSWORD" macos_codesigning.keychain
security default-keychain -s macos_codesigning.keychain
security unlock-keychain -p "$APPLE_CODESIGNING_PASSWORD" macos_codesigning.keychain
security import macos_codesigning.p12 -k macos_codesigning.keychain -P "$APPLE_CODESIGNING_PASSWORD" -T /usr/bin/codesign
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$APPLE_CODESIGNING_PASSWORD" macos_codesigning.keychain

codesign -s "$APPLE_CODESIGNING_ID" -v --deep --timestamp -o runtime dist/DistanceEstimation.app -f

ditto -c -k --keepParent dist/DistanceEstimation.app dist/DistanceEstimation.zip

xcrun notarytool submit dist/DistanceEstimation.zip --wait --apple-id "$APPLE_NOTARIZATION_USER_ID" --team-id "$APPLE_NOTARIZATION_TEAM_ID" --password "$APPLE_NOTARIZATION_PASSWORD"

xcrun stapler staple dist/DistanceEstimation.app