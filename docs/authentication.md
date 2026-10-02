# GENROGRAM — Authentication & Security Architecture

## Authentication Providers

- **Firebase Authentication:**
  - Email & Password
  - Google OAuth Sign-In
  - Authenticator TOTP MFA Support (Prepared)

## Security Rules

- **Firestore Collections:**
  - `users`: Accessible only by authenticated owner.
  - `classification_history`: Queryable only by `userId == auth.uid`. User predictions isolated.
  - `user_preferences`: Owner read/write.
