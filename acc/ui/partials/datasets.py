"""Dataset browser and generator partials."""

from starlette.requests import Request
from starlette.responses import HTMLResponse

from acc.ui.api import call as _api


async def partial_datasets(request: Request):
    """Dataset browser with sample thumbnails."""
    datasets = await _api("/datasets")
    if not datasets or (isinstance(datasets, dict) and "error" in datasets):
        return HTMLResponse(
            '<div class="panel"><h3>Datasets</h3><div class="empty">No datasets</div></div>'
        )

    items = ""
    for ds in datasets:
        name = ds["name"]
        size = ds["size"]
        shape = "x".join(str(d) for d in ds["image_shape"])
        target_type = ds.get("target_type", "?")
        items += f"""
        <div style="margin-bottom:10px;border-bottom:1px solid #21262d;padding-bottom:8px;">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#f0f6fc;font-weight:600;">{name}</span>
                <span style="color:#8b949e;font-size:11px;">{size} images</span>
            </div>
            <div style="color:#8b949e;font-size:11px;">{shape} | target: {target_type}</div>
            <div id="ds-samples-{name}" style="margin-top:4px;"
                hx-get="/partial/dataset_samples/{name}"
                hx-trigger="load"
                hx-swap="innerHTML">
            </div>
        </div>"""

    return HTMLResponse(
        f'<div class="panel"><h3>Datasets</h3>{items if items else "<div class=empty>No datasets</div>"}</div>'
    )


async def partial_dataset_samples(request: Request):
    """Sample thumbnails for a dataset."""
    name = request.path_params["name"]
    samples = await _api(f"/datasets/{name}/sample?n=8")
    if samples and "images" in samples:
        imgs = "".join(
            f'<img src="data:image/png;base64,{b64}" style="width:32px;height:32px;image-rendering:pixelated;border:1px solid #30363d;">'
            for b64 in samples["images"]
        )
        return HTMLResponse(f'<div style="display:flex;gap:2px;flex-wrap:wrap;">{imgs}</div>')
    return HTMLResponse('<span style="color:#484f58;font-size:10px;">No samples</span>')


async def partial_generate(request: Request):
    """[+ Dataset] panel — pick generator, configure params, generate."""
    generators = await _api("/registry/generators")

    if not generators or isinstance(generators, dict):
        return HTMLResponse(
            '<div class="panel"><h3>+ Dataset</h3><div class="empty">No generators available</div></div>'
        )

    gen_options = '<option value="">-- select generator --</option>'
    param_forms = ""
    for gen in generators:
        gen_name = gen["name"]
        gen_desc = gen.get("description", "")
        gen_options += f'<option value="{gen_name}" title="{gen_desc}">{gen_name}</option>'

        params = gen.get("parameters", {})
        param_html = ""
        for pname, pinfo in params.items():
            ptype = pinfo.get("type", "text")
            pdefault = pinfo.get("default", "")
            pdesc = pinfo.get("description", "")
            input_type = "number" if ptype == "int" or ptype == "float" else "text"
            param_html += f"""
            <div class="form-group">
                <label>{pname}: {pdesc}</label>
                <input type="{input_type}" name="param_{pname}" value="{pdefault}">
            </div>"""

        param_forms += f'<div id="gen-params-{gen_name}" class="gen-params" style="display:none;">{param_html}</div>'

    return HTMLResponse(f"""
    <div class="panel">
        <h3>+ Dataset</h3>
        <div class="form-group">
            <label>Generator</label>
            <select name="generator_name" id="gen-select"
                onchange="document.querySelectorAll('.gen-params').forEach(e=>e.style.display='none'); var sel=this.value; if(sel) document.getElementById('gen-params-'+sel).style.display='block';">
                {gen_options}
            </select>
        </div>
        {param_forms}
        <button class="btn btn-primary" style="width:100%;margin-top:4px;"
            hx-post="/action/generate_dataset"
            hx-include="#gen-select, [name^=param_]"
            hx-target="#generate-panel"
            hx-swap="innerHTML">Generate</button>
    </div>
    """)
