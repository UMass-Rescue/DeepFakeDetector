from typing import TypedDict
from flask_ml.flask_ml_server.models import BatchTextInput, ResponseBody, BatchTextResponse, TextResponse

class TransformCaseInputs(TypedDict):
    text_inputs: BatchTextInput

class TransformCaseParameters(TypedDict):
    to_case: str # 'upper' or 'lower'

@server.route("/transform_case")
def transform_case(inputs: TransformCaseInputs, parameters: TransformCaseParameters) -> ResponseBody:
    to_upper: bool = parameters['to_case'] == 'upper'
    
    outputs = []
    for text_input in inputs['text_inputs'].texts:
        raw_text = text_input.text
        processed_text = raw_text.upper() if to_upper else raw_text.lower()
        outputs.append(TextResponse(value=processed_text, title=raw_text))

    return ResponseBody(root=BatchTextResponse(texts=outputs))