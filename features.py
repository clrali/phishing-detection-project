from __future__ import annotations
from dataclasses import dataclass
from typing import List
from bs4 import BeautifulSoup

@dataclass
class HTMLFeatures:
    """Extracts a fixed-length numeric feature vector from an HTML document."""

    soup: BeautifulSoup

    def _has_any(self, tag: str) -> int:
        return int(bool(self.soup.find(tag)))

    def _count(self, tag: str) -> int:
        return len(self.soup.find_all(tag))

    def _count_inputs_by_type(self, input_type: str) -> int:
        return sum(1 for inp in self.soup.find_all("input")
                   if (inp.get("type") or "").lower() == input_type)

    def _has_input_with_attr_value(self, value: str) -> int:
        for inp in self.soup.find_all("input"):
            attrs = [
                (inp.get("type") or "").lower(),
                (inp.get("name") or "").lower(),
                (inp.get("id") or "").lower()
            ]
            if value in attrs:
                return 1
        return 0

    def to_vector(self) -> List[int | float]:
        s = self.soup

        # simple boolean flags
        has_title        = int(bool(s.title and s.title.text.strip()))
        has_input        = int(bool(s.find("input")))
        has_button       = int(bool(s.find("button")))
        has_image        = int(bool(s.find("img") or s.find("image")))
        has_submit       = int(self._count_inputs_by_type("submit") > 0)
        has_link         = int(bool(s.find("link")))
        has_password     = self._has_input_with_attr_value("password")
        has_email_input  = self._has_input_with_attr_value("email")
        has_hidden_input = self._count_inputs_by_type("hidden") > 0
        has_audio        = self._has_any("audio")
        has_video        = self._has_any("video")

        # numeric counts
        num_inputs       = self._count("input")
        num_buttons      = self._count("button")
        num_imgs_html    = self._count("img") + self._count("image")
        num_options      = self._count("option")
        num_list_items   = self._count("li")
        num_th           = self._count("th")
        num_tr           = self._count("tr")
        num_paragraphs   = self._count("p")
        num_scripts      = self._count("script")
        num_a            = self._count("a")
        num_div          = self._count("div")
        num_figure       = self._count("figure")
        num_meta         = self._count("meta")
        num_source       = self._count("source")
        num_span         = self._count("span")
        num_table        = self._count("table")

        # links in <link> tags
        num_href = sum(1 for link in s.find_all("link") if link.get("href"))

        # title + text length
        title_len = len(s.title.text) if s.title else 0
        text_len  = len(s.get_text())

        # headings
        has_h1 = int(bool(s.find("h1")))
        has_h2 = int(bool(s.find("h2")))
        has_h3 = int(bool(s.find("h3")))

        # clickable buttons (type="button")
        num_click_buttons = self._count_inputs_by_type("button") + sum(
            1 for b in s.find_all("button")
            if (b.get("type") or "").lower() == "button"
        )

        # layout-ish tags
        has_footer   = self._has_any("footer")
        has_form     = self._has_any("form")
        has_textarea = self._has_any("textarea")
        has_iframe   = self._has_any("iframe")
        has_text_inp = int(self._count_inputs_by_type("text") > 0)
        has_nav      = self._has_any("nav")
        has_object   = self._has_any("object")
        has_picture  = self._has_any("picture")

        return [
            has_title,
            has_input,
            has_button,
            has_image,
            int(has_submit),
            has_link,
            has_password,
            has_email_input,
            int(has_hidden_input),
            has_audio,
            has_video,
            num_inputs,
            num_buttons,
            num_imgs_html,
            num_options,
            num_list_items,
            num_th,
            num_tr,
            num_href,
            num_paragraphs,
            num_scripts,
            title_len,
            has_h1,
            has_h2,
            has_h3,
            text_len,
            num_click_buttons,
            num_a,
            num_imgs_html,
            num_div,
            num_figure,
            has_footer,
            has_form,
            has_textarea,
            has_iframe,
            has_text_inp,
            num_meta,
            has_nav,
            has_object,
            has_picture,
            num_source,
            num_span,
            num_table,
        ]


def extract_features_from_html(html: str) -> list[int | float]:
    """Convenience wrapper: HTML text -> feature vector."""
    soup = BeautifulSoup(html, "html.parser")
    return HTMLFeatures(soup).to_vector()

