"""Child drawing psychology knowledge base and theoretical references.

Grounds the analysis engine in established developmental psychology and
projective art therapy frameworks:
- John N. Buck (1948): House-Tree-Person (HTP) projective drawing technique
- Elizabeth M. Koppitz (1968): Emotional Indicators on Human Figure Drawings (HFD)
- Viktor Lowenfeld (1947/1987): Creative and Mental Growth / developmental stages
- Cathy A. Malchiodi (1998): Understanding Children's Drawings
- Esther Burkitt, Martyn Barrett & Alyson Davis (2003-2005): Affective color & size studies
- Rose H. Alschuler & La Berta W. Hattwick (1947): Painting and Personality
- Max Pulver: Spatial symbolism in projective graphology/drawing
"""

from typing import Dict, List, Any

EMOTION_KEYS = [
    "sadness",
    "anger",
    "anxiety",
    "loneliness",
    "joy",
    "calm",
]

EMOTION_PROFILES: Dict[str, Dict[str, Any]] = {
    "sadness": {
        "key": "sadness",
        "label": "傷心與低落",
        "short_label": "傷心低落",
        "en_label": "Sadness & Melancholy",
        "valence": "內收・低能量",
        "accent_hex": "#4A78A8",
        "summary": "畫面透露出較為沉重、緩慢或內收的情緒氛圍，可能正經歷委屈、失落、疲憊，或有說不出口的難過。",
        "inner_voices": [
            "「心裡好像有點沉重，提不起勁，只想安靜待在屬於自己的小角落。」",
            "「有些委屈或難過卡在心裡，不知道該怎麼開口，只好透過顏色和線條慢慢流出來。」",
            "「我覺得有點累，好希望能被溫柔抱抱，不需要馬上堅強也沒關係。」",
        ],
        "care_prompts": [
            "這幅畫裡哪一個地方感覺最重或最冷？如果可以替它加上一樣溫暖的東西，你會想畫什麼？",
            "如果畫裡的世界正在下小雨，畫面中的主角最希望現在誰能陪在身邊？",
            "畫完這張畫之後，心裡有沒有比剛剛輕鬆了一點點？",
        ],
    },
    "anger": {
        "key": "anger",
        "label": "憤怒與挫敗",
        "short_label": "憤怒挫敗",
        "en_label": "Anger & Frustration",
        "valence": "外放・高張力",
        "accent_hex": "#D9483B",
        "summary": "筆觸與色彩展現高度能量與張力，可能蘊含生氣、挫折、不被理解的煩躁，或是急切想要衝破束縛、宣洩壓力的衝動。",
        "inner_voices": [
            "「我心裡塞滿了氣和不甘心，好想用力把這些煩躁全部畫出來、甩出去！」",
            "「我覺得很不公平或不被尊重，有好多話想大聲喊出來卻沒人聽見。」",
            "「內在的能量快要滿出來了，我正在找一個安全的出口釋放這股張力。」",
        ],
        "care_prompts": [
            "這些最有力的線條，如果發出聲音的話，聽起來會像雷聲、吼叫，還是什麼聲音？",
            "最近有沒有一件事讓你覺得『好生氣』或『好不公平』，很想幫自己說句話？",
            "如果這股強大的力量能變成一種超能力，你想用它來保護什麼？",
        ],
    },
    "anxiety": {
        "key": "anxiety",
        "label": "焦慮與壓力",
        "short_label": "焦慮壓力",
        "en_label": "Anxiety & Tension",
        "valence": "緊繃・防禦性",
        "accent_hex": "#C88A2E",
        "summary": "線條呈現較多停頓、塗抹、反覆修正或包圍緊縮感，常對應內心的擔憂、緊張、對犯錯的害怕，或面對環境變化的不安。",
        "inner_voices": [
            "「我心裡緊繃繃的，一直在想這樣做對不對、會不會不夠好。」",
            "「腦袋裡有好多纏在一起的念頭，不知道該從哪裡整理起，有點害怕出錯。」",
            "「我需要多一點界線和安全感，確定自己是被保護的、不會突然受傷。」",
        ],
        "care_prompts": [
            "在畫畫的過程中，哪一段讓你最猶豫或想重新修改？那時候心裡在擔心什麼呢？",
            "如果能為這幅畫留出一塊讓人深深吐一口氣的安全空間，你會放在哪裡？",
            "最近有什麼事情讓你覺得壓力好大、好像一直鬆懈不下來？",
        ],
    },
    "loneliness": {
        "key": "loneliness",
        "label": "孤單與渴望連結",
        "short_label": "孤單渴望",
        "en_label": "Loneliness & Seeking Connection",
        "valence": "疏離・期盼陪伴",
        "accent_hex": "#6E65A3",
        "summary": "畫面留白比例偏高、元素較小或彼此疏離，反映出孤單感、覺得自己被忽略，內心深處其實渴望被看見、被理解與建立連結。",
        "inner_voices": [
            "「周圍看起來好寬、好安靜，我好像一個人待在邊邊，有點希望有人能走過來看看我。」",
            "「我想靠近大家，但又有點不確定自己是不是受歡迎，所以先靜靜待在原地。」",
            "「其實我心裡藏著一塊柔軟的地方，好想和懂我的人分享此刻的感受。」",
        ],
        "care_prompts": [
            "畫裡這個小小的空間看起來很安靜，如果可以邀請一個朋友或動物進來，你想邀請誰？",
            "你覺得畫裡面的人或主角，現在最想聽到別人對他說哪一句話？",
            "在日常生活中，什麼時候最會讓你覺得『有人懂我、陪著我』？",
        ],
    },
    "joy": {
        "key": "joy",
        "label": "愉悅與活力",
        "short_label": "愉悅活力",
        "en_label": "Joy & Vitality",
        "valence": "開展・正向能量",
        "accent_hex": "#E09128",
        "summary": "明亮的暖色系與舒展流暢的筆觸顯示出充沛的心理能量、好奇心與正向情感，內心處於開放、樂於表達與分享的狀態。",
        "inner_voices": [
            "「現在心裡亮亮的、很有活力，想把眼前好玩或溫暖的感覺自由地畫出來！」",
            "「我覺得自己可以自在表現，對周圍的世界充滿好奇與期待。」",
            "「我想分享這份開心的能量，讓看到這幅畫的人也能感受到一點陽光。」",
        ],
        "care_prompts": [
            "這幅畫裡你最喜歡哪一個部分？畫到那裡的時候心裡想到了什麼開心的事？",
            "如果這幅畫是一個可以走進去玩的世界，你最想在裡面做什麼？",
            "你會想把這幅畫分享給誰看？想跟對方介紹畫裡的哪個小秘密？",
        ],
    },
    "calm": {
        "key": "calm",
        "label": "平靜與安全感",
        "short_label": "平靜安全",
        "en_label": "Calm & Groundedness",
        "valence": "穩定・和諧平衡",
        "accent_hex": "#3B8C6E",
        "summary": "色彩和諧、構圖穩重居中且線條連貫從容，反映出內在情緒平穩、具有安全感，處於能專注沉澱與自我安頓的狀態。",
        "inner_voices": [
            "「現在的心境很安穩，不急不躁，正用自己舒服的節奏感受這個當下。」",
            "「我覺得這裡是安全的，可以慢慢整理思緒，享受專注創作的寧靜。」",
            "「雖然生活有起伏，但此刻我心裡有一個平靜穩固的小基地。」",
        ],
        "care_prompts": [
            "這幅畫散發出很安靜舒服的感覺，是什麼讓你現在能這麼平靜專注？",
            "畫面裡哪一個顏色或形狀最能代表你此刻的『安心感』？",
            "當以後遇到忙碌或煩躁的日子，你會想如何把今天這份平靜帶回身邊？",
        ],
    },
}

PSYCHOLOGY_REFERENCES: List[Dict[str, str]] = [
    {
        "id": "buck_htp",
        "author": "John N. Buck (1948)",
        "title": "屋樹人繪畫投射測驗（House-Tree-Person, HTP）",
        "focus": "空間構圖、畫面比例、封閉邊界與象徵意義",
        "description": "提出畫面位置（中央、偏下、角落、邊界溢出）、圖形大小與塗抹修正，可反映個體的安全感、內外在界線與心理能量狀態。",
    },
    {
        "id": "koppitz_hfd",
        "author": "Elizabeth M. Koppitz (1968)",
        "title": "兒童人物畫情緒指標（Emotional Indicators on HFDs）",
        "focus": "重度塗黑陰影、極小構圖、斷續線條、高頻擦除",
        "description": "系統整理 30 項兒童繪畫情緒指標，指出重度塗抹/陰影與反覆塗改常與焦慮有關，極小縮在角落的構圖常對應退縮、低落與不安全感。",
    },
    {
        "id": "lowenfeld",
        "author": "Viktor Lowenfeld (1947 / 1987)",
        "title": "創造力與心智成長（Creative and Mental Growth）",
        "focus": "兒童藝術發展階段與線條情緒表現力",
        "description": "探討線條流暢度、銳利轉折、重複刻板圖樣與自我表達自由度如何隨情感狀態與認知發展而變化。",
    },
    {
        "id": "burkitt_color",
        "author": "Esther Burkitt, Martyn Barrett & Alyson Davis (2003–2005)",
        "title": "兒童繪畫中色彩與尺寸的情感表徵研究",
        "focus": "暖明色 vs. 暗沉/非偏好色彩與主題尺寸效應",
        "description": "實證研究指出兒童在描繪正向情感主題時傾向畫得較大且選用偏愛/明亮色彩；面對負向情緒主題時則顯著增加暗色、低飽和度色彩與縮放變化。",
    },
    {
        "id": "malchiodi",
        "author": "Cathy A. Malchiodi (1998)",
        "title": "讀懂孩子的畫（Understanding Children's Drawings）",
        "focus": "藝術治療視角、整體脈絡與自發性口語敘說",
        "description": "強調單一筆觸或顏色不可斷章取義，應綜合色彩、線條、構圖與創作者自身的口語故事（Spontaneous Verbalization），以溫柔提問開啟對話而非貼標籤。",
    },
    {
        "id": "alschuler_hattwick",
        "author": "Rose H. Alschuler & La Berta W. Hattwick (1947)",
        "title": "繪畫與人格（Painting and Personality）",
        "focus": "線條動力學（曲線 vs. 尖角）與色彩喚起度",
        "description": "觀察幼兒與兒童的色彩與線條偏好，發現強烈紅黑搭配與急促尖角常關聯高喚起衝動或挫敗感，而圓弧與平衡色調較多出現於情緒和諧時。",
    },
]

# Spontaneous verbalization lexicon (Traditional & Simplified Chinese + English)
VERBAL_LEXICON: Dict[str, List[str]] = {
    "sadness": [
        "難過", "傷心", "想哭", "哭", "眼淚", "淚", "委屈", "失落", "沮喪", "低落",
        "下雨", "雨天", "烏雲", "黑雲", "好累", "疲憊", "心痛", "心碎", "不開心",
        "難受", "悲傷", "失望", "痛", "sad", "cry", "tears", "rain", "upset", "hurt",
    ],
    "anger": [
        "生氣", "憤怒", "氣死", "討厭", "煩", "煩躁", "不爽", "爆炸", "火大", "罵",
        "吵架", "打架", "不公平", "恨", "吼", "怒", "抓狂", "暴風雨", "雷電", "怪物",
        "angry", "mad", "hate", "furious", "annoyed", "storm", "fight",
    ],
    "anxiety": [
        "緊張", "焦慮", "害怕", "怕", "擔心", "壓力", "考試", "功課", "迷宮", "亂",
        "做不好", "犯錯", "恐懼", "噩夢", "黑", "發抖", "不安", "不知所措", "喘不過氣",
        "anxious", "scared", "fear", "worry", "stress", "nervous", "maze",
    ],
    "loneliness": [
        "孤單", "寂寞", "一個人", "沒人", "沒有人", "不想說話", "躲起來", "被忽略",
        "沒朋友", "想家", "離開", "空空的", "角落", "邊緣", "孤獨", "陪我",
        "lonely", "alone", "nobody", "empty", "miss", "isolated",
    ],
    "joy": [
        "開心", "快樂", "高興", "好玩", "喜歡", "愛", "陽光", "太陽", "花", "彩虹",
        "笑", "微笑", "朋友", "慶祝", "禮物", "幸福", "棒", "溫暖", "自由", "飛",
        "happy", "joy", "fun", "sun", "rainbow", "smile", "love", "play",
    ],
    "calm": [
        "平靜", "安靜", "舒服", "放鬆", "安心", "安全", "溫柔", "家", "森林", "海邊",
        "星星", "月亮", "休息", "睡覺", "微風", "和平", "穩定", "靜靜",
        "calm", "peace", "quiet", "safe", "relax", "home", "gentle", "rest",
    ],
}

# Optional user-tagged motif weights (HTP / KFD / Malchiodi symbolic themes)
USER_MOTIF_WEIGHTS: Dict[str, Dict[str, float]] = {
    "sun": {"joy": 0.14, "calm": 0.06},
    "rain_cloud": {"sadness": 0.16, "loneliness": 0.08, "anxiety": 0.05},
    "monster_storm": {"anger": 0.16, "anxiety": 0.12},
    "house": {"calm": 0.08, "loneliness": 0.04},
    "tree": {"calm": 0.08, "joy": 0.07},
    "person": {"loneliness": 0.05, "joy": 0.04},
    "animal": {"joy": 0.08, "calm": 0.07, "loneliness": 0.04},
    "abstract": {"anxiety": 0.04, "anger": 0.04},
}
