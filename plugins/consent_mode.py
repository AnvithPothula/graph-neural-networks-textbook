"""MkDocs hook: inject Google Consent Mode v2 defaults on every page.

Must run before the gtag/js script loads, so we inject the consent
initialization block immediately before the Google Tag Manager <script>
tag that MkDocs Material emits when analytics.provider = google.

Consent defaults for this site (educational, non-commercial, no ads):
  analytics_storage  = granted  -- basic visit/engagement tracking is fine
  ad_storage         = denied   -- no advertising cookies needed
  ad_user_data       = denied   -- no user data shared with ad platforms
  ad_personalization = denied   -- no ad audience building

This silences the four "consent signals inactive" warnings in GA4 and
satisfies Google Consent Mode v2 without requiring a consent banner.
"""

_CONSENT_SCRIPT = (
    "<script>\n"
    "  window.dataLayer = window.dataLayer || [];\n"
    "  function gtag(){dataLayer.push(arguments);}\n"
    "  gtag('consent', 'default', {\n"
    "    'analytics_storage': 'granted',\n"
    "    'ad_storage': 'denied',\n"
    "    'ad_user_data': 'denied',\n"
    "    'ad_personalization': 'denied'\n"
    "  });\n"
    "</script>"
)

_ANALYTICS_MARKER = '<script id="__analytics">'


def on_post_page(output, page, config, **kwargs):
    """Inject consent defaults immediately before Material's analytics block."""
    if _ANALYTICS_MARKER in output:
        output = output.replace(
            _ANALYTICS_MARKER,
            _CONSENT_SCRIPT + "\n" + _ANALYTICS_MARKER,
            1,  # replace first occurrence only
        )
    return output
