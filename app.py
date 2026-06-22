from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/euro_ncap_model.joblib')

FEATURES = [
    'kerb_weight',
    'seat1_knee_airbag',
    'row2_side_chest_airbag',
    'seat1_side_pelvis_airbag',
    'seat3_side_pelvis_airbag',
    'row2_side_pelvis_airbag',
    'seat1_centre_lateral_airbag',
    'seat3_centre_lateral_airbag',
    'seat3_isofix',
    'seat1_isize',
    'seat3_isize',
    'row2_isize',
    'row3_isize',
    'seat3_childdetection',
    'row2_childdetection',
    'has_active_bonnet',
    'has_distraction_detection',
    'has_fatigue_detection',
    'has_lane_assist',
    'has_speed_assist',
    'has_aeb_vru',
    'has_cyclistdoorprevention',
    'has_aeb_m2w',
    'has_aeb_cartocar',
    'protocol_version'
]

BOOL_FIELDS = [
    'seat1_knee_airbag',
    'row2_side_chest_airbag',
    'seat1_side_pelvis_airbag',
    'seat3_side_pelvis_airbag',
    'row2_side_pelvis_airbag',
    'seat1_centre_lateral_airbag',
    'seat3_centre_lateral_airbag',
    'seat3_isofix',
    'seat1_isize',
    'seat3_isize',
    'row2_isize',
    'row3_isize',
    'seat3_childdetection',
    'row2_childdetection',
    'has_active_bonnet',
    'has_distraction_detection',
    'has_fatigue_detection',
    'has_lane_assist',
    'has_speed_assist',
    'has_aeb_vru',
    'has_cyclistdoorprevention',
    'has_aeb_m2w',
    'has_aeb_cartocar'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        try:
            values = []
            for feature in FEATURES:
                raw_value = request.form.get(feature)
                if raw_value is None:
                    raise ValueError(f'Missing input for {feature}')
                if feature == 'kerb_weight':
                    values.append(float(raw_value))
                elif feature == 'protocol_version':
                    values.append(int(raw_value))
                else:
                    values.append(1 if raw_value == 'yes' else 0)

            X = np.array([values])
            prediction = model.predict(X)[0]
        except Exception as exc:
            error = str(exc)

    return render_template('index.html', features=FEATURES, bool_fields=BOOL_FIELDS, prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
