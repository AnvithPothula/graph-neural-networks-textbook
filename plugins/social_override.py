"""MkDocs hook that injects Open Graph and Twitter Card meta tags into
every rendered page using the page's frontmatter.

Loaded via the `hooks:` entry in mkdocs.yml, not as a plugin -- this
avoids collisions with other projects that also install a package
called `social_override` or a top-level module called `plugins`.

Frontmatter fields read (per page):
    title         -- falls back to the H1/page.title
    description   -- falls back to site_description
    image         -- path under docs/ (e.g. img/cover.png); falls back to
                     site-wide default `img/cover.png`

The hook emits absolute URLs by joining `site_url` with the image path,
so the og:image / twitter:image tags work for crawlers and the
bk-check-social-cover verifier (which HEAD-requests the image URL).

If the social plugin (mkdocs-material[imaging]) is enabled, this hook
also replaces its auto-generated /assets/images/social/ tags with the
declared cover image.
"""

import re

DEFAULT_IMAGE = "img/cover.png"


def _absolute_image_url(site_url, image_path):
    if image_path.startswith(("http://", "https://")):
        return image_path
    return site_url.rstrip("/") + "/" + image_path.lstrip("/")


def _meta_tag(prop, content, attr="property"):
    return f'<meta {attr}="{prop}" content="{content}">'


def on_post_page(html, page, config, **kwargs):
    site_url = config.get("site_url") or ""
    if not site_url:
        return html

    title = (page.meta or {}).get("title") or page.title or config.get("site_name", "")
    description = (page.meta or {}).get("description") or config.get("site_description", "")
    image = (page.meta or {}).get("image") or DEFAULT_IMAGE
    image_url = _absolute_image_url(site_url, image)

    # Replace social-plugin-emitted /assets/images/social/ tags first.
    og_social = re.compile(
        r'<meta\s+property="og:image"\s+content="[^"]*/assets/images/social/[^"]*"[^>]*>'
    )
    html = og_social.sub(_meta_tag("og:image", image_url), html)

    tw_social = re.compile(
        r'<meta\s+(?:property|name)="twitter:image"\s+content="[^"]*/assets/images/social/[^"]*"[^>]*>'
    )
    html = tw_social.sub(_meta_tag("twitter:image", image_url, attr="name"), html)

    # Inject any tags that are still missing, immediately before </head>.
    injections = []
    if not re.search(r'<meta\s+property="og:title"', html):
        injections.append(_meta_tag("og:title", title))
    if not re.search(r'<meta\s+property="og:description"', html):
        injections.append(_meta_tag("og:description", description))
    if not re.search(r'<meta\s+property="og:image"', html):
        injections.append(_meta_tag("og:image", image_url))
    if not re.search(r'<meta\s+property="og:type"', html):
        injections.append(_meta_tag("og:type", "website"))
    if not re.search(r'<meta\s+property="og:url"', html):
        page_url = site_url.rstrip("/") + "/" + (page.url or "").lstrip("/")
        injections.append(_meta_tag("og:url", page_url))

    if not re.search(r'<meta\s+(?:property|name)="twitter:card"', html):
        injections.append(_meta_tag("twitter:card", "summary_large_image", attr="name"))
    if not re.search(r'<meta\s+(?:property|name)="twitter:title"', html):
        injections.append(_meta_tag("twitter:title", title, attr="name"))
    if not re.search(r'<meta\s+(?:property|name)="twitter:description"', html):
        injections.append(_meta_tag("twitter:description", description, attr="name"))
    if not re.search(r'<meta\s+(?:property|name)="twitter:image"', html):
        injections.append(_meta_tag("twitter:image", image_url, attr="name"))

    if injections:
        html = html.replace("</head>", "  " + "\n  ".join(injections) + "\n</head>", 1)

    return html
