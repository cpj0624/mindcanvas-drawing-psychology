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
    # --- Peer-Reviewed International Journal Empirical Studies (v3.0) ---
    {
        "id": "jolley_2004",
        "category": "國際期刊實驗",
        "author": "Jolley, R. P., Fenn, K., & Jones, L. (2004)",
        "journal": "British Journal of Developmental Psychology, 22(4), 545–567",
        "doi": "10.1348/0261510042378236",
        "title": "兒童表達性繪畫的發展實驗（Literal / Content / Abstract）",
        "focus": "情境內容優先於純抽象形式、區分畫技與情緒表達力",
        "description": "實證顯示兒童情緒繪畫隨年齡成長主要體現在「情境內容與符號（Content）」的豐富化而非抽象形式，且寫實畫技與情緒表達力僅呈弱相關（畫得簡略不代表情緒負面）。",
    },
    {
        "id": "picard_2007",
        "category": "國際期刊實驗",
        "author": "Picard, D., Brechet, C., & Baldy, R. (2007)",
        "journal": "Journal of Nonverbal Behavior, 31(4), 243–257",
        "doi": "10.1007/s10919-007-0035-5",
        "title": "繪畫表達策略與年齡、主題之實驗（N = 120）",
        "focus": "人物偏好字面臉部表情、房屋偏好非字面結構與情境線索",
        "description": "對 7–8 歲、11–12 歲兒童與成人對照實驗發現：繪製人物時優先採字面臉部表情（L），繪製房屋則優先採非字面情境／結構（NLC/NLA），隨年齡增長傾向併用多種策略。",
    },
    {
        "id": "picard_2012",
        "category": "國際期刊實驗",
        "author": "Picard, D., & Gauthier, C. (2012)",
        "journal": "Child Development Research, 2012, Article 925063",
        "doi": "10.1155/2012/925063",
        "title": "兒童至青少年屋樹人表達性繪畫大樣本常模（N = 480，5–15 歲）",
        "focus": "5–10 歲字面為主 → 11–15 歲複合隱喻；單色媒材自動以形狀替代色彩",
        "description": "480 名兒童與青少年實證指出，7 歲起即能穩定表達情緒（失敗率 < 2%），且當媒材僅限單色時會自然改用形狀與情境內容替代色彩，直接支持「單色鉛筆去偏」與分齡發展常模。",
    },
    {
        "id": "misalidi_2014",
        "category": "國際期刊實驗",
        "author": "Misalidi, P., & Bonoti, F. (2014)",
        "journal": "Early Child Development and Care",
        "doi": "10.1080/03004430.2013.823409",
        "title": "情緒誘發對兒童繪畫表達策略之影響實驗（N = 96）",
        "focus": "跨年齡情境內容（Content）使用頻率顯著高於抽象形式（Abstract）",
        "description": "在 5、7、9、11 歲正向／負向情緒誘發後繪製人與樹的實驗中，證實所有年齡層運用「情境隱喻」皆顯著多於純抽象色塊線條，支持圖形情境高於純色彩權重。",
    },
    {
        "id": "tanaka_2020",
        "category": "國際期刊實驗",
        "author": "Nanayama Tanaka, C., Higa, H., Ogawa, N., Ishido, M., Nakamura, T., & Nishiwaki, M. (2020)",
        "journal": "Frontiers in Psychology, 11, 576683 (PMC7773925)",
        "doi": "10.3389/fpsyg.2020.576683",
        "title": "負向情緒與臉部表情繪畫嘴部幾何量化實驗（FACED × POMS）",
        "focus": "嘴部寬度、圓度、開口封閉笑 vs. 上凸下彎弧／極短直線／省略嘴巴",
        "description": "電腦幾何分析證實：正向心情組顯著畫出較寬、高圓度、封閉開口笑或下凹弧線（U 型）嘴巴；高負向情緒組顯著傾向「上凸下垂弧（∩ 型）」、「較短直線」或「省略嘴巴（無口）」。",
    },
    {
        "id": "burkitt_2003a",
        "category": "國際期刊實驗",
        "author": "Burkitt, E., Barrett, M., & Davis, A. (2003a)",
        "journal": "Journal of Child Psychology and Psychiatry, 44(3), 445–455",
        "doi": "10.1111/1469-7610.00134",
        "title": "情感主題與兒童色彩偏好選擇實驗（N = 330，4–11 歲）",
        "focus": "正向用個人偏愛色、負向用最不喜愛色；不可一對一綁死單色與情緒",
        "description": "330 名兒童實驗顯示兒童會依「個人色彩偏好」映射情感角色（負向常選黑色），但研究強調解讀用色必須考量個人喜好，不可將單一顏色僵化對應單一情緒。",
    },
    {
        "id": "burkitt_2003b_2004",
        "category": "國際期刊實驗",
        "author": "Burkitt, E., Barrett, M., & Davis, A. (2003b; 2004)",
        "journal": "British Journal of Developmental Psychology, 21(4), 565–583; Educational Psychology, 24(3), 315–343",
        "doi": "10.1348/026151003322535228 / 10.1080/0144341042000211670",
        "title": "情感特質對兒童自由繪畫尺寸之不對稱效應實驗（N = 258 / N = 253）",
        "focus": "正向穩定放大、高威脅不一定縮小，僅退縮／孤單明顯縮小",
        "description": "在有範本與無範本自由繪畫實驗中皆證實：兒童畫正向角色會穩定放大尺寸，但畫負向或具威脅性角色時不一定縮小，需區分「大且舒展（正向）」與「大且尖銳重塗（威脅張力）」。",
    },
    {
        "id": "crawford_2012",
        "category": "國際期刊實驗",
        "author": "Crawford, E., Gross, J., Patterson, T., & Hayne, H. (2012)",
        "journal": "Infant and Child Development, 21(2), 198–215",
        "doi": "10.1002/icd.742",
        "title": "兒童真實情感事件自由繪畫之寫實原色實驗",
        "focus": "寫實原色（藍天、綠草、黃太陽）強烈主導選色，降權色彩誤判",
        "description": "要求兒童自由繪製真實開心／傷心事件，發現選色受寫實物件原色（Canonical Color）強烈支配，色彩與情緒強度幾乎無關（r = -0.14, n.s.），避免將藍天綠地誤判為憂鬱。",
    },
    {
        "id": "penzes_2020",
        "category": "國際期刊實驗",
        "author": "Pénzes, I., van Hooren, S., Dokter, D., Smeijsters, H., & Hutschemaekers, G. (2020)",
        "journal": "Frontiers in Psychology, 11, 572700 (PMC7546898)",
        "doi": "10.3389/fpsyg.2020.572700",
        "title": "藝術作品形式元素與心理健康及心理彈性實證研究",
        "focus": "動態、流動、輪廓、重複之結構平衡，兼顧正向適應資源",
        "description": "四階段質性與量化實證證實「動態、流動、輪廓、重複」四大形式元素的結構平衡顯著關聯心理彈性（AAQ-II）與適應力，而色彩僅為次要調整項；臨床上須同時看見創作者的正向資源。",
    },
    {
        "id": "joiner_1996_catte_1999",
        "category": "國際期刊實驗",
        "author": "Joiner, T. E. Jr., Schmidt, K. L., & Barnett, J. (1996); Catte, M., & Cox, M. V. (1999)",
        "journal": "Journal of Personality Assessment, 67(1), 127–141; European Child & Adolescent Psychiatry, 8(2), 86–91",
        "doi": "10.1207/s15327752jpa6701_10 / 10.1007/s007870050089",
        "title": "投射指標單獨效度檢驗（N = 80 臨床樣本與 Koppitz HFD 實證）",
        "focus": "多證據收斂護欄：單一線條粗重、尺寸或單一指標不足以推斷困擾",
        "description": "實證顯示單獨使用尺寸、細節量、線條輕重或單一 Koppitz 指標與臨床憂鬱焦慮關聯極低，因此本系統強制採用「多維證據收斂」門檻並在低線索時主動降低極端分數。",
    },
    {
        "id": "thomas_jolley_1998",
        "category": "國際期刊實驗",
        "author": "Thomas, G. V., & Jolley, R. P. (1998)",
        "journal": "British Journal of Clinical Psychology, 37(2), 127–139",
        "doi": "10.1111/j.2044-8260.1998.tb01289.x",
        "title": "兒童繪畫心理評估之實證與概念重審",
        "focus": "多重成因與替代性溫和解釋、畫作為多源證據之一",
        "description": "指出同一繪畫特徵可能源自畫技、空間規劃習慣或個人風格而非情緒困擾，繪畫應作為多源收斂線索之一並誠實呈現替代可能解釋。",
    },
    {
        "id": "gross_hayne_1998",
        "category": "國際期刊實驗",
        "author": "Gross, J., & Hayne, H. (1998)",
        "journal": "Journal of Experimental Psychology: Applied, 4(2), 163–179",
        "doi": "10.1037/1076-898X.4.2.163",
        "title": "繪畫促進兒童情緒事件口語敘說對照實驗",
        "focus": "邊畫邊說使真實情緒細節倍增；開放式、非誘導性提問原則",
        "description": "兩項實驗證實讓兒童「邊畫邊說」能比純口語訪談多說出兩倍以上的情緒經驗細節且不降低準確度，核心前提是陪伴者採用開放式、不預設立場的非誘導提問。",
    },
    {
        "id": "cao_2023",
        "category": "國際期刊實驗",
        "author": "Cao, H., Zhang, X., Zhang, X., & Yan, W. (2023)",
        "journal": "Frontiers in Psychology / PMC10726032 (PMID: 38111865)",
        "doi": "PMID: 38111865",
        "title": "華語文化脈絡下情緒與情境啟動對人樹畫尺寸與色彩之實驗（N = 113）",
        "focus": "華語「孤單」顯著縮小偏冷色、「平和」具正向舒展、高威脅反擴大偏紅",
        "description": "以華語樣本檢驗平和、感恩與孤單情緒的人樹畫，證實孤單畫作顯著更小且偏冷色，「平和」具正向效價且幅面舒展，而高威脅情境反而畫幅擴大且紅色比例上升。",
    },
    # --- Classic Foundational Projective & Developmental Frameworks ---
    {
        "id": "classic_foundations",
        "category": "經典理論架構",
        "author": "Buck (1948); Koppitz (1968); Lowenfeld (1987); Malchiodi (1998); Alschuler & Hattwick (1947)",
        "journal": "HTP 屋樹人 / HFD 人物畫情緒指標 / 藝術治療基礎文獻",
        "doi": "Classic Foundations",
        "title": "兒童繪畫心理學五大經典投射與藝術發展架構",
        "focus": "屋樹人門窗空間、30 項 HFD 指標、線條發展階段與藝術治療陪伴倫理",
        "description": "構成空間九宮格、房屋門窗界線、重度陰影與反覆擦除等經典符號基礎，並結合上述現代國際期刊實驗進行去偏與多證據收斂校準。",
    },
]

AGE_GROUP_NORMS: Dict[str, Dict[str, Any]] = {
    "unspecified": {
        "label": "未指定（通用模式）",
        "expectedStyle": "綜合字面表情、情境內容與形式結構",
        "omittedDetailScale": 1.0,
        "guidance": "未提供創作者年齡，採用通用圖形結構優先與多證據收斂常模。",
    },
    "3-6": {
        "label": "3–6 歲（學齡前）",
        "expectedStyle": "字面表情與簡易幾何圖式（蝌蚪人階段）",
        "omittedDetailScale": 0.45,
        "guidance": "依 Picard & Gauthier (2012) 與 Jolley et al. (2004)，3–6 歲筆觸簡略、省略細節或比例誇張屬正常動作認知發展，系統已自動降低單一省略／尺寸權重。",
    },
    "7-10": {
        "label": "7–10 歲（國小中低年級）",
        "expectedStyle": "字面表情＋情境隱喻並行期",
        "omittedDetailScale": 0.85,
        "guidance": "依 Picard et al. (2007; 2012)，7–10 歲已具備穩定情緒表達力（失敗率 < 2%），常以人物表情搭配太陽、雨雲、房屋門窗等情境線索傳達心情。",
    },
    "11+": {
        "label": "11 歲以上（高年級／青少年／成人）",
        "expectedStyle": "字面＋情境＋抽象形式複合表達期",
        "omittedDetailScale": 1.0,
        "guidance": "依 Picard & Gauthier (2012)，11 歲以上逐漸增加非字面隱喻與形式張力（構圖留白、線條動態、結構重複）來表達細微且多層次的情緒。",
    },
}

TOOL_HARDNESS: Dict[str, float] = {
    "marker": 1.0,
    "crayon": 0.78,
    "pencil": 0.48,
    "watercolor": 0.35,
}

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
