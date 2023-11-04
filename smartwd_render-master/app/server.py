import aiohttp
import asyncio
import uvicorn
import numpy as np
import cv2
from fastai import *
from fastai.vision.all import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

path = '/'
def get_x(r): return path+'/images_original/'+r['image'] # create path to open images in the original folder
def get_y(r): return r['label']



export_file_url = 'https://www.googleapis.com/drive/v3/files/13JJNR8zRZjUzPWhkSrXywjR9YH0Vpl16?alt=media&key=AIzaSyCYGkKHllanXFFoNxZJ1jcjwpgBCVJWev8'
export_file_name = 'model_v4.pkl'
classes = ["T-Shirt","Longsleeve","Pants","Shirt","Dress","Outwear","Shorts","Not_sure","Hat","Skirt","Polo","Undershirt","Blazer","Hoodie","Thawb","Body","Top","Blouse"]
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(str(path) + "/" + export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb
def get_image_color(image):
    img = cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)

    height, width = img.shape[:2]
    # Change these values to fit the size of your region of interest
    roi_size = 10 # (10x10)
    roi_values = img[(height-roi_size)//2:(height+roi_size)//2,(width-roi_size)//2:(width+roi_size)//2]
    mean_blue = np.mean(roi_values[:,:,2])
    mean_green = np.mean(roi_values[:,:,1])
    mean_red = np.mean(roi_values[:,:,0])
    
    color = "#"+rgb_to_hex((int(mean_blue), int(mean_green), int(mean_red)))
    
    return color


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = Image.open(BytesIO(img_bytes))
    color = get_image_color(img)
    img = np.array(img)
    pred,pred_idx,probs = learn.predict(img)
    prediction = pred
    return JSONResponse({'result': str(prediction) , 'color' : str(color)})


if __name__ == '__main__': 
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
