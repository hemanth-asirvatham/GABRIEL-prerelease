"""Prompt template utilities."""

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, Optional, Set

from jinja2 import Environment, PackageLoader, meta

from ..utils.jinja import shuffled, shuffled_dict


@dataclass
class PromptTemplate:
    """Simple Jinja2-based prompt template."""

    text: str

    def render(self, **params: Dict[str, str]) -> str:
        """Render the template with the given parameters."""
        attrs = params.get("attributes")
        descs = params.get("descriptions")
        if isinstance(attrs, list):
            if isinstance(descs, list) and len(descs) == len(attrs):
                params["attributes"] = {a: d for a, d in zip(attrs, descs)}
            else:
                params["attributes"] = {a: a for a in attrs}
        env = Environment(loader=PackageLoader("gabriel", "prompts"))
        env.filters["shuffled_dict"] = shuffled_dict
        env.filters["shuffled"] = shuffled
        template = env.from_string(self.text)
        return template.render(**params)

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_variables(text: str) -> Set[str]:
        """Return the set of undeclared variables in ``text``.

        A minimal Jinja environment is used that registers the filters
        expected by the built-in prompts.  This allows templates using
        those filters to be parsed without error while ignoring their
        actual behaviour.
        """

        env = Environment()
        env.filters["shuffled_dict"] = lambda x: x
        env.filters["shuffled"] = lambda x: x
        ast = env.parse(text)
        return meta.find_undeclared_variables(ast)

    @classmethod
    def from_file(
        cls,
        path: str,
        *,
        reference_filename: Optional[str] = None,
        package: str = "gabriel.prompts",
    ) -> "PromptTemplate":
        """Load a template from ``path`` with optional variable checking.

        If ``reference_filename`` is provided, the custom template is
        validated to ensure it declares the exact same set of variables
        as the reference template.  A descriptive ``ValueError`` is
        raised if there is a mismatch.
        """

        text = Path(path).read_text(encoding="utf-8")
        if reference_filename is not None:
            ref_text = resources.files(package).joinpath(reference_filename).read_text(
                encoding="utf-8"
            )
            vars_custom = cls._extract_variables(text)
            vars_ref = cls._extract_variables(ref_text)
            missing = vars_ref - vars_custom
            extra = vars_custom - vars_ref
            if missing or extra:
                parts = []
                if missing:
                    parts.append(f"missing variables: {sorted(missing)}")
                if extra:
                    parts.append(f"unexpected variables: {sorted(extra)}")
                raise ValueError("Custom template variable mismatch (" + "; ".join(parts) + ")")
        return cls(text)

    @classmethod
    def from_package(
        cls,
        filename: str,
        package: str = "gabriel.prompts",
    ) -> "PromptTemplate":
        """Load a template from the given package file."""
        text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        return cls(text)
