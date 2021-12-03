import cv2
from handdetector import HandDetector
import av
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
#import queue #decomment all queues to have it on the website
from PIL import Image
import hydralit_components as hc
#from bokeh.models.widgets import Div
from model_utils import build_model

# Import component
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

#add_fronta
app_formal_name = "🧙 Sign detection 🐍"

# Start the app in wide-mode

st.set_page_config(
    layout="wide", page_title=app_formal_name,
)

#navbar

# specify the primary menu definition
menu_data = [
        {'icon': "fas fa-video", 'label':"Webcam"},
        {'icon':"fas fa-hand-sparkles",'label':"Sign learning"},
        {'icon': "fas fa-bus", 'label':"Le Wagon"}, #can add a tooltip message
        {'icon': "fas fa-users", 'label':"Teammates"},
        {'icon': "fab fa-youtube", 'label':"Demo Day"},
]
# we can override any part of the primary colors of the menu
over_theme = {'txc_inactive': '#FFFFFF', 'menu_background' : '#2176ae'}
menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home',override_theme=over_theme)

##########################################
######### navbar menu Le Wagon ###########
##########################################

if menu_id == "Le Wagon":
    col1, col2, empty_col = st.columns([8,1,1])
    col2.write("<h1 style='text-align: center;'></h1><br><br>", unsafe_allow_html=True)
    img_logo = Image.open("./images/logo_wagon.JPG")
    img_logo2 = Image.open("./images/logo_sld.JPG")
    col2.image(img_logo)
    col2.image(img_logo2)

    img_wagon = Image.open("./images/img_code.JPG")
    col1.image(img_wagon)
    info2 = """
    <br>
    <h3 style='text-align: center;'>Through immersive coding bootcamps!</h3>
    <br>
    """
    col1.write(info2, unsafe_allow_html=True)

    # m = col1.markdown("""
    # <style>
    # div.stButton > button:first-child {
    #     background-color: #e63946;
    #     color:#ffffff;
    # }
    # div.stButton > button:hover {
    #     background-color: #f77f00;
    #     color:#ffffff;
    #     }
    # </style>""", unsafe_allow_html=True)

    # link = '[Le Wagon](https://www.lewagon.com/fr)'
    # if col1.button(label="Go to LeWagon Website!"):
    #         js = "window.open('https://www.lewagon.com/fr')"  # New tab or window
    #         html = '<img src onerror="{}">'.format(js)
    #         div = Div(text=html)
    #         st.bokeh_chart(div)


##########################################
######### navbar menu Teammates ##########
##########################################

if menu_id == "Teammates":
    empty1, col1,empty2, col2,empty3 ,col3, empty4 = st.columns([0.6,0.4,0.6,0.9,0.6,0.4,0.6])
    col1.markdown( "[![this is an image link](https://img.icons8.com/nolan/2x/github.png)](https://github.com/Dannxs)")
    col1.markdown("<p style='text-align: center;'><b>Danny Cardoso</p>", unsafe_allow_html=True)
    col1.markdown("<h2 style='text-align: center;'>Follow us!</h2>", unsafe_allow_html=True)
    col3.markdown( "[![this is an image link](https://img.icons8.com/nolan/2x/github.png)](https://github.com/jvesp)")
    col3.markdown("<p style='text-align: center;'><b>Julien Vesperini</p>", unsafe_allow_html=True)
    col3.markdown("<h2 style='text-align: center;'>Favorite Team!</h2>", unsafe_allow_html=True)
    img_follow = Image.open("./images/follow_us.JPG")
    col2.image(img_follow)
    col1.markdown( "[![this is an image link](https://img.icons8.com/nolan/2x/github.png)](https://github.com/glauret)")
    col1.markdown("<p style='text-align: center;'><b>Guillaume Lauret</p>", unsafe_allow_html=True)
    col3.markdown( "[![this is an image link](https://img.icons8.com/nolan/2x/github.png)](https://github.com/selmalopez)")
    col3.markdown("<p style='text-align: center;'><b>Selma Lopez</p>", unsafe_allow_html=True)

##########################################
####### navbar menu Sign Learning ########
##########################################

if menu_id == "Sign learning":
    col1, col2, col3 = st.columns([4,5,4])
    img_signs = Image.open("./images/img_sign_main.JPG")
    col2.image(img_signs)

##########################################
########### navbar menu Home #############
##########################################

if menu_id == "Home":

    img = Image.open("./images/hands.JPG")
    st.image(img)
    col1, col2 = st.columns(([14,1]))
    #info
    info = '''
    <p>Made with 💙 by <a href='https://github.com/jvesp/sld'>Detection language team</a></p>'''.strip()
    
    info2="""
    [![Star](https://img.shields.io/github/stars/jvesp/sld.svg?logo=github&style=social)](https://github.com/jvesp/sld)
    """
    col1.write(info, unsafe_allow_html=True)
    col2.write(info2, unsafe_allow_html=True)

##########################################
########## navbar menu Webcam ############
##########################################

if menu_id == "Webcam":
    empty_left, col2, empty_right = st.columns([0.5, 1 , 0.5])

    #dictionary of traduction letters
    dict_letter = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'K', 10:'L', 11:'M',
                12:'N', 13:'O', 14:'P', 15:'Q', 16:'R', 17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y' }
    COLORS = np.random.uniform(0, 255, size=(len(dict_letter), 3))
    dict_colors = {}
    for i in range(24):
        dict_colors[i] = COLORS[i]

    #Set up STUN servers
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # HERE = Path(__file__).parent
    # print(os.listdir(HERE))
    # if not 'model.h5' in os.listdir(HERE):
    #     txt = st.warning("Téléchargement du modèle")
    #     print("loading model")
    #     url = 'https://www.dropbox.com/s/sffb5ew98us9gxa/model_resnet50_V2_8830.h5?dl=1'
    #     u = urllib.request.urlopen(url)
    #     data = u.read()
    #     u.close()
    #     with open('model.h5', 'wb') as f:
    #         f.write(data)
    #     print("model loaded")
    #     txt.success("Téléchargement terminé")

    @st.cache(allow_output_mutation=True)
    def load_mo():
        #model = load_model('models/model_resnet50_V2_8830.h5')
        model = build_model()
        model.load_weights("models/model_resnet50_V2_8830_weights.h5")
        return model

    # Your class where you put the intelligence
    class SignPredictor(VideoProcessorBase):

        def __init__(self) -> None:
            # Hand detector
            self.hand_detector = HandDetector(detectionCon=0.8, maxHands=1)
            self.model = load_mo()
            self.counter = 0
            self.l=[]
            self.word=[]
            #self.result_queue_word = queue.Queue()

        def find_hands(self, image):

            #add the rectangle in your image around the hands
            hands, image_hand = self.hand_detector.findHands(image)

            if hands:
                bbox1 = hands[0]["bbox"]  # Bounding box info x,y,w,h
                x, y, w, h = bbox1

                #récup img plus grande que la bbox1 de base
                x_square = int(x - 0.2 * w)
                y_square = int(y - 0.2 * h)
                w_square = int(x + 1.2 * w)
                h_square = int(y + 1.2 * h)

                #anticipe erreur de x, y négatifs
                if x_square < 0 :
                    x_square = 0
                if y_square < 0:
                    y_square = 0
                if h_square < 0:
                    h_square = 0
                if w_square < 0:
                    w_square = 0

                hand_img = image_hand[y_square:h_square,x_square:w_square]  # image of the hand
                img_hand_resize = np.array(
                    tf.image.resize_with_pad(hand_img, 256, 256))  # resize image to match model's expected sizing
                img_hand_resize = img_hand_resize.reshape(1, 256, 256, 3)
                img_hand_resize = tf.math.divide(img_hand_resize, 255)

                #couleur img_main
                channels = tf.unstack(img_hand_resize, axis=-1)
                img_hand_resize = tf.stack([channels[2], channels[1], channels[0]],
                                        axis=-1)

                prediction = self.model.predict(img_hand_resize)[0]

                probabs = round(prediction[np.argmax(prediction)], 2)
                pred = np.argmax(prediction)


                self.counter +=1
                if self.counter % 1 == 0:

                    if probabs > 0.8:
                        self.l.append(dict_letter[pred])

                    # COLORING BOX
                    cv2.rectangle(image_hand, (x_square, y_square),
                                  (w_square, h_square), (dict_colors[pred]), 2)

                    # Finds space required by the text so that we can put a background with that amount of width.
                    (w, h), _ = cv2.getTextSize(f'{dict_letter[pred]} - {str(probabs)}',
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

                    # Prints the box, letter and proba
                    img = cv2.rectangle(image_hand, (x_square, y_square - 20),
                                        (x_square + w, y_square),
                                        (dict_colors[pred]), -1)
                    img = cv2.putText(image_hand,
                                      f"{dict_letter[pred]} - {str(probabs)}",
                                      (x_square, y_square - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.6, (0,0,0), 2)

                    # WORD CREATION
                    if len(self.l) == 15:
                        self.word.append(max(set(self.l), key=self.l.count))
                        self.l = []
                    final_word = ""
                    for letters in self.word:
                        final_word = final_word + letters

                    # Draw rectangle    p1(x,y)    p2(x,y)    Student name box

                    # cv2.rectangle(image_hand, (195, 55), (250, 80), (42, 219, 151), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(image_hand, final_word, (200, 60), font, 1.5,
                                (20, 1, 170), 4)

            else:
                if self.word:
                    if self.word[-1] != " ":
                        self.word.append(" ")
                        #self.result_queue_word.put(self.word)


            return hands, image_hand

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            hands, annotated_image = self.find_hands(image)
            return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    with col2:
        webrtc_ctx = webrtc_streamer(
                key="object-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=SignPredictor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
