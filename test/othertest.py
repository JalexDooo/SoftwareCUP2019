import requests
import json
import re

class urlinf:
    def __init__(self):
        self.url_gettoken = "http://47.100.1.234:8085/VMW/api/GetToken"
        self.url_postwarning = "http://47.100.1.234:8085/VMW/api/PostWarnInfo"
        self.parambox = {'code': 'C001', 'key': '123'}
        self.payload = {
                        "CameraCode": "C003",
                        "ImageURL": "\\Content\\Files\\WarningImage\\video.jpg",
                        "ImageName": "video",
                        "VideoURL": "\\Content\\Files\\WarningVideo\\test_mv02.mov",
                        "VideoName": "test_mv02",
                        "DES": "人员未佩戴安全帽",
                        "CategoryCode": "/",
                        "CategoryName": "/",
                        "LevelCode": "/",
                        "LevelName": "/",
                        "WarningDate": "2019-02-22 10:10:10.000"
                    }

        self.jsondata = requests.get(self.url_gettoken, params=self.parambox)
        self.token = self.jsondata.text[self.jsondata.text.find('}') - 17:self.jsondata.text.find('}') - 1]
        self.payload["token"] = self.token

    def gettoken(self):
        # re.match()
        print(self.jsondata.text.find('}'))
        print(self.token, self.jsondata.text)
        return self.token

    def postwarning(self):
        s = requests.session()
        post = s.post(self.url_postwarning, json=self.payload)
        print(post.text)
        return post


if __name__ == "__main__":
    urltext = urlinf()
    urltext.gettoken()
    urltext.postwarning()