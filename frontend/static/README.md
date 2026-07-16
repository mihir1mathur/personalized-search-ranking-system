# frontend/static

Static assets served alongside the Streamlit app (e.g. a logo, favicon, or
downloadable files). Kept as a dedicated folder so static content is separate
from generated diagrams/screenshots (which live in `frontend/assets/`).

The current UI styling is provided by `../styles.css`; images used in the app
and README are generated into `../assets/` by `python frontend/generate_assets.py`.
