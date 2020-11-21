import keyboard
from PIL import ImageGrab, Image
import win32clipboard as cb
from io import BytesIO
import requests
import base64

import winsound
import time

class CropperClient:

    def __init__(self, api_url, shortcut, path):
        self.api_url = api_url
        self.shortcut = shortcut
        self.path = path
        
    def start(self):
        keyboard.add_hotkey(self.shortcut, self.send_cropping_request)
        keyboard.wait('esc')

    def prepare_image_data(self):
        img = ImageGrab.grabclipboard()
        buff = BytesIO()
        img.save(buff, format="PNG")
        img_b64 = base64.b64encode(buff.getvalue())
        return img_b64

    def send_cropping_request(self):
        image_data = self.prepare_image_data()
        
        content = { "img": image_data,
                    "settings": { "is_url": False,
                                  "preprocessing": False, 
                                  "postprocessing": False,
                                  "model": "u2net"} }
                                 
        print("sending request...")
        start_time = time.time()
        try:
            r = requests.post(f"{self.api_url}/cropping-api/crop", timeout=30.0, json=content)
            if r.status_code == 200:
                print(f"[+] successful response : response time {round(time.time()-start_time,3)}")
                res_image = Image.open(BytesIO(r.content))
                self.save_cropped_image(img_data=res_image)
            else:
                print("[-] unsuccessful response")
        except Exception as e:
            print(f"[-] exception in sending request : {e}")

    def save_cropped_image(self, img_data):
        img_data.save(f"{self.path}/cropped_img_{int(time.time())}.png", 'PNG')

        buff = BytesIO()
        img_data.convert('RGB').save(buff, 'BMP')
        data = buff.getvalue()[14:]
        buff.close()
        cb.OpenClipboard()
        cb.EmptyClipboard()
        cb.SetClipboardData(cb.CF_DIB, data)
        cb.CloseClipboard()
        
        winsound.Beep(frequency=2000, duration=200)
        print("[+] saved cropped image & copied to clipboard (not transparent)\n-------------")
        

if __name__ == "__main__":
    c = CropperClient(api_url="http://127.0.0.1:5000", shortcut="alt+x", path="C:/Users/msi/Projects/cropper/cropper-service/client/win-desktop-python/cropped_imgs")
    c.start()
