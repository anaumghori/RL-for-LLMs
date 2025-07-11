import json, os, io
import torch
from transformers import AutoModelForCausalLM
from bottle import request
import bottle, threading, queue


os.environ['TOKENIZERS_PARALLELISM'] = 'true'
model_path = "Qwen/Qwen3-8B"

reference_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16,        
    _attn_implementation="sdpa").to("cuda") 

reference_model.eval()
reference_model.requires_grad_(False)


def get_token_prob(input_ids):
    """
    computes the log-probability assigned by the model to each token in the input, conditioned on the 
    tokens that came before it. this is useful for evaluating how confident the model is about its own 
    predictions at each position in the sequence.
    """
    logits = reference_model(input_ids).logits  # Shape: [batch size, sequence length, vocab size]

    # exclude the last token's prediction (it doesn't predict anything)
    logits = logits[:, :-1, :]  # Shape: [B, L-1, V]

    # exclude the first input ID since we don't have logits for it
    input_ids = input_ids[:, 1:]  # Shape: [B, L-1]
    
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_logps = logits_row.log_softmax(dim=-1)  # Shape: [L-1, V]

        # select the log-prob of the actual next token at each position
        token_log_prob = torch.gather(
            log_logps, dim=1,
            index=input_ids_row.unsqueeze(1)  # Shape: [L-1, 1]
        ).squeeze(1)  # Shape: [L-1]
        
        per_token_logps.append(token_log_prob)

    # combine all results into a single tensor
    return torch.stack(per_token_logps)  # Shape: [B, L-1]


def tensor_to_bytes(t):
    buffer = io.BytesIO() # in-memory buffer
    torch.save(t, buffer) # save tensors to buffer
    byte_data = buffer.getvalue()
    return byte_data

def bytes_to_tensor(b):
    buffer = io.BytesIO(b) # wrap the raw bytes into a buffer
    tensor = torch.load(buffer, weights_only=True) # load the tensor from this buffer
    return tensor

def make_bytes_list(blist):
    buffer = io.BytesIO()
    # write the number of items in the list
    num_items = len(blist) 
    buffer.write(num_items.to_bytes(4, 'big'))
    for b in blist:
        item_length = len(b) # length of byte string
        buffer.write(item_length.to_bytes(4, 'big')) # write the length 
        buffer.write(b) # write the actual byte string

    # get the final packed binary blob
    packed_data = buffer.getvalue()
    return packed_data

def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    # read the first 4 bytes = number of items
    num_items = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num_items):
        # read the next 4 bytes to get the length of this item
        item_length = int.from_bytes(buffer.read(4), 'big')
        # read that many bytes (the actual item)
        item_bytes = buffer.read(item_length)
        blist.append(item_bytes)
    return blist # return the reconstructed list

if __name__ == '__main__':
    raw_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()
    app = bottle.Bottle()
    @app.route('/upload', method='POST')
    def do_upload():
        dd = request.body.read()
        dd = bytes_list_to_list(dd)
        if len(dd) not in (3,4): return b'tensor'
        data = {'base': json.loads(dd[0])} 
        data['inputs'] = bytes_to_tensor(dd[1])
        data['rewards'] = bytes_to_tensor(dd[2])
        if len(dd) == 4: data['gen_logps'] = bytes_to_tensor(dd[3])
        raw_queue.put(data)
        return b'tensor'

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty(): return b'empty'
        return result_queue.get()
    
    def run_server(): bottle.run(app, host='0.0.0.0', port=59875, server='tornado')
    threading.Thread(target=run_server, daemon=False).start()

    while True:
        d = raw_queue.get()
        prompt_length = d['base']['plen']
        with torch.inference_mode():
            per_token_logps = get_token_prob(d['inputs'].to(reference_model.device))
        per_token_logps = per_token_logps[:,prompt_length-1:]
        data = [json.dumps(d['base']).encode(), tensor_to_bytes(d['inputs']), 
                tensor_to_bytes(d['rewards']), tensor_to_bytes(per_token_logps)]
        if 'gen_logps' in d: data.append(tensor_to_bytes(d['gen_logps']))
        xdata = make_bytes_list(data)
        result_queue.put(xdata)