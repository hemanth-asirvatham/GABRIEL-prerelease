"""Prompt template utilities."""

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
import warnings
from typing import Dict, Optional, Set

from jinja2 import Environment, PackageLoader, Template, meta

from ..utils.jinja import shuffled, shuffled_dict


@dataclass
class PromptTemplate:
    """Simple Jinja2-based prompt template."""

    text: str
    _environment: Environment = field(init=False, repr=False)
    _template: Template = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._environment = Environment(loader=PackageLoader("gabriel", "prompts"))
        self._environment.filters["shuffled_dict"] = shuffled_dict
        self._environment.filters["shuffled"] = shuffled
        self._template = self._environment.from_string(self.text)

    def render(self, **params: Dict[str, str]) -> str:
        """Render the template with the given parameters."""
        attrs = params.get("attributes")
        descs = params.get("descriptions")
        if isinstance(attrs, list):
            if isinstance(descs, list) and len(descs) == len(attrs):
                params["attributes"] = {a: d for a, d in zip(attrs, descs)}
            else:
                params["attributes"] = {a: a for a in attrs}
        return self._template.render(**params)

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
                msg = "Custom template variable mismatch (" + "; ".join(parts) + ")"
                required = {v for v in vars_ref if v in {"text", "attributes"}}
                missing_required = required - vars_custom
                if missing_required:
                    raise ValueError(msg)
                warnings.warn(
                    msg + "; proceeding because required variables are present.",
                    UserWarning,
                    stacklevel=2,
                )
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


def resolve_template(
    *,
    template: Optional[PromptTemplate],
    template_path: Optional[str],
    reference_filename: str,
    package: str = "gabriel.prompts",
) -> PromptTemplate:
    """Return a prompt template using either an object or a filesystem override.

    Parameters
    ----------
    template:
        Optional :class:`PromptTemplate` instance supplied directly by the caller.
    template_path:
        Filesystem path to a custom Jinja2 template.  When provided, the template
        is validated against ``reference_filename`` to ensure it exposes the same
        variables as the built-in prompt.
    reference_filename:
        Name of the packaged template used as the default for the task.  This is
        used both as the fallback when no overrides are supplied and as the
        reference for variable validation when ``template_path`` is set.
    package:
        Package containing the built-in prompt templates.  Callers rarely need to
        override this.
    """

    if template is not None and template_path is not None:
        raise ValueError("Provide either template or template_path, not both")

    if template_path is not None:
        template = PromptTemplate.from_file(
            template_path,
            reference_filename=reference_filename,
            package=package,
        )

    return template or PromptTemplate.from_package(
        reference_filename,
        package=package,
    )
