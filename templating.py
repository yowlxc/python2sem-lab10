from fastapi.responses import HTMLResponse
from mako.lookup import TemplateLookup
from pathlib import Path

# Путь к папке с шаблонами (где лежит index.mako)
TEMPLATES_DIR = Path("templates")

template_lookup = TemplateLookup(
    directories=[str(TEMPLATES_DIR)],
    input_encoding='utf-8',
    output_encoding='utf-8',
    default_filters=['h'],
)

def render_template(template_name: str, **context):
    """Рендерит Mako шаблон и возвращает HTMLResponse"""
    template = template_lookup.get_template(template_name)
    rendered = template.render(**context)
    return HTMLResponse(content=rendered)