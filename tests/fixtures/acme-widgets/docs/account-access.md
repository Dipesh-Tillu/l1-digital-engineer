category: account/access

# Account Access

## Password resets

A customer locked out of their account can trigger a password reset email
from the login page; reset links expire after 30 minutes. If a customer
says the reset email never arrived, check the account's email-on-file for
a typo before assuming a delivery problem - mistyped emails at signup are
the most common cause.

## Two-factor lockouts

A customer who lost access to their 2FA device cannot self-serve a reset -
this requires identity verification (billing zip + last 4 of the payment
method on file) before support can disable 2FA on the account. Never
disable 2FA on a verbal claim alone.

## Account takeover reports

If a customer reports their account was accessed by someone else (orders
they didn't place, changed shipping address), treat this as
account/access and lock the account immediately pending investigation -
do not wait for identity verification to complete before locking, since
the lock itself is the containment step and verification determines who
gets access back.

## Merging duplicate accounts

Acme does not support merging two customer accounts; if a customer created
a second account by mistake, order history stays split across both and
support should record which account is "primary" going forward rather
than attempting a merge.
