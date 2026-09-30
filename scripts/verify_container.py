"""Run inside the app image via stdin; verify real inference and manual UI.

PowerShell: Get-Content -Raw scripts/verify_container.py | docker exec -i NAME python -
Linux: docker exec -i NAME python < scripts/verify_container.py
No extra test dependencies, training data, credentials, or language service needed.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from streamlit.testing.v1 import AppTest

from src.config import ROOT
from src.interface import REQUIRED_FEATURES, SAMPLE_FEATURES
from src.serving import load_serving_model, predict, training_defaults


def main():
    assert ROOT == Path('/app'), 'Run this check inside the app container.'
    assert os.getuid() == 10001, 'The application must run as its non-root user.'
    assert not (ROOT / '.env').exists()
    assert not (ROOT / '.tools').exists()
    with urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5) as response:
        assert response.read().decode().strip() == 'ok'

    model, metadata = load_serving_model()
    defaults = training_defaults(model)
    core = {key: SAMPLE_FEATURES[key] for key in REQUIRED_FEATURES}
    full_price = predict(model, metadata, SAMPLE_FEATURES)
    core_price = predict(model, metadata, core)
    assert abs(full_price - 159062.25) < .01
    assert abs(core_price - 160000.625) < .01
    assert core_price == predict(model, metadata, {**defaults, **core})

    # Exercise actual Streamlit state with the saved model; no model mocks.
    os.environ.update(AMES_LLM_ENABLED='false', AMES_HOSTED='false', AMES_DEMO_PASSWORD='')
    app = AppTest.from_file(str(ROOT / 'app/streamlit_app.py'), default_timeout=60).run()
    assert not app.exception and app.button(key='estimate').disabled
    for key, value in core.items():
        if isinstance(value, str):
            app.selectbox(key='field_' + key).select(value)
        else:
            app.text_input(key='field_' + key).set_value(str(value))
    app.run()
    assert app.button(key='estimate').disabled
    app.checkbox(key='confirm_details').check().run()
    app.button(key='estimate').click().run()
    first = app.session_state['home'].result
    assert first['price'] == core_price and len(first['defaults']) == 10
    assert any(item.value == '$160,001' for item in app.metric)

    app.text_input(key='field_Garage Cars').set_value('0').run()
    assert app.session_state['home'].stale
    assert app.session_state['home'].result == first
    assert not app.checkbox(key='confirm_details').value
    assert app.button(key='estimate').disabled
    app.checkbox(key='confirm_details').check().run()
    app.button(key='estimate').click().run()
    corrected = app.session_state['home'].result
    assert corrected['features']['Garage Cars'] == 0
    assert 'Garage Cars' not in corrected['defaults']
    assert corrected['price'] == predict(model, metadata, {**core, 'Garage Cars': 0})

    app.text_input(key='field_Garage Cars').set_value('-1').run()
    assert app.button(key='estimate').disabled and app.error
    app.text_input(key='field_Garage Cars').set_value('').run()
    app.checkbox(key='confirm_details').check().run()
    app.button(key='estimate').click().run()
    restored = app.session_state['home'].result
    assert 'Garage Cars' in restored['defaults'] and restored['price'] == core_price
    assert not app.exception

    files = sorted([*ROOT.glob('src/*.py'), *ROOT.glob('configs/*'),
                    ROOT / 'app/requirements.txt', ROOT / 'app/streamlit_app.py',
                    ROOT / '.streamlit/config.toml'])
    report = {
        'verified_at': datetime.now(timezone.utc).isoformat(),
        'passed': True, 'linux_uid': os.getuid(), 'health': 'ok',
        'env_file_in_image': False, 'local_llm_weights_in_image': False,
        'housing_model_sha256': metadata['sha256'], 'housing_model_run': metadata['run_id'],
        'full_example_prediction': full_price, 'four_field_prediction': core_price,
        'zero_garage_prediction': corrected['price'],
        'checks': {
            'real_released_model': True, 'partial_equals_explicit_defaults': True,
            'manual_ui_prediction_visible': True, 'confirmation_required': True,
            'correction_marks_previous_estimate': True, 'zero_preserved': True,
            'invalid_optional_blocks_estimate': True, 'unknown_restores_default': True,
            'no_ui_exceptions': True,
        },
        'application_files_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in files if path.is_file()},
        'scope': 'Built Linux image, Streamlit health, real released-model inference and manual review/correction flow. Language inference is verified separately.',
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
