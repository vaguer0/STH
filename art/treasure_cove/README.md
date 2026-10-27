# Kayıp Korsan Koyu — SeekThenHide

Hazine avı için küçük bir harita: yaklaşık 55 m genişliğinde tropik bir ada. Low-poly,
düz gölgeli (flat shaded) stil. Her şey `build_treasure_cove.py` ile Blender içinde üretilir.

![hero](renders/cove_hero.png)

| Tepeden | Hazine haritası |
| --- | --- |
| ![top](renders/cove_top.png) | ![map](renders/treasure_map.png) |

## Dosyalar

| Dosya | Ne işe yarar |
| --- | --- |
| `export/treasure_cove.glb` | Oyuna atılacak dosya (arazi, yapılar, bitki örtüsü, spawn noktaları) |
| `treasure_cove.blend` | Düzenlenebilir sahne; ışıklar ve kameralar `Preview_Rig` koleksiyonunda |
| `renders/treasure_map.png` | Parşömen tarzı yerleşim haritası (numaralı hazine noktaları, patikalar) |
| `renders/treasure_props_map.png` | Aynı harita, 200 eşyanın yerleriyle |
| `renders/props_catalog.png` | Bütün eşya çeşitleri yan yana, isimleriyle |
| `renders/cove_*.png` | Önizlemeler: genel görünüm, tapınak, doğu ve batı yakası, tepeden |

## Ada

Oyuncular doğudaki **iskeleden** başlar; patikalar iskeleden, kamptan ve batık gemiden
tepedeki tapınakta birleşir.

- **Tapınak (merkez tepe):** üç katlı taş platform, güneyde merdiven ve iki meşale, arka
  sırada lento taşıyan üç sütun, önde kırık sütunlar, ortada sunak. Arkasında devrik
  sütunlar ve dağınık taşlar.
- **Batık gemi (güney koyu):** kumsala oturmuş, yatık bir korsan gemisi; gövdesinde yarık,
  kırık güverte, yırtık yelkenli kırık direk. Kumsalda dağılmış fıçı ve sandıklar.
- **İskele (doğu):** 11 m uzunluğunda, iki tahtası eksik ahşap iskele ve bağlı bir sandal.
- **Kaya kemeri (kuzeybatı tepesi):** iki kaya sütun ve üstünde yosunlu bir taş; altı küçük
  bir mağara gibi saklanmaya uygun.
- **Korsan kampı (batı palmiyeliği):** kırmızı yelken bezinden çadır, kamp ateşi, kütük
  banklar, fıçılar, sandıklar ve tapınağı gösteren bir tabela.
- **Adacık (kuzeydoğu açığı):** iki palmiyeli küçük bir kum adası. Ana adadan diz boyu
  (-0,45 m) bir kum seti üzerinden yürünerek gidilir.
- **Deniz feneri (doğu kıyısı):** kırmızı-beyaz çizgili taş kule. Dışındaki spiral merdivenden
  korkuluklu balkona çıkılır; tepede parlayan ışık odası var.
- **Balıkçı kulübesi (güneydoğu kumsalı):** kazıklar üzerinde, sazdan çatılı tahta kulübe;
  verandası, merdiveni, pencereleri, yanında ağ kurutma iskelesi ve ters çevrilmiş bir sandal.
- **Yıkık taş ev (güneydoğu çayırı):** bir köşesi çökmüş taş duvarlar, kapı ve pencere
  boşlukları, içeri devrilmiş çatı kirişleri ve moloz.
- **Kuyu (iskele patikası):** taş bilezik, çatılı makara ve asılı kova.
- **Mezarlık (doğu tepesi):** kapılı çitle çevrili; mezar taşları, tahta haçlar, açık bir
  mezar, toprak yığını, kürek ve kurumuş bir ağaç.
- **Gözcü kulesi (güneybatı):** 4 m yüksekliğinde ahşap kule; merdiveni, korkuluklu
  platformu, sazdan çatısı ve bayrağı var.
- **Kaçakçı barakası (batı kumsalı):** önü açık tahta baraka, içinde fıçı ve kasa yığınları.
- Yeni yapıların hepsine patika gider.
- Saklanmak için iki düzine kadar palmiye, ~25 çalı ve ~25 kaya; kuzey kumsalında arkasına saklanılabilecek
  büyük kayalar.

## Detay

- Kutular pah kırılmış kenarlı; yuvarlak parçalar yumuşak, keskin kenarlar sert gölgelenir
  (köşe normalleri glb'de).
- **Tapınak:** katlar tek tek taş bloklardan örülü (blok boyu ve tonu değişken, derzli),
  üstleri döşeme taşlı, yer yer yosunlu; sütunlar yivli, entasisli, kaideli ve başlıklı,
  kırık olanların uçları kırık; lento bloklu, dişli korniş, sarkan sarmaşıklar, kabartmalı sunak.
- **Palmiyeler:** halka izli, tabanı genişleyen gövde; her yaprak dalı tek tek yaprakçıklı,
  kuru sarkan yapraklar ve hindistan cevizi salkımları. Çalılar çok loblu, bir kısmı çiçekli.
- **Batık gemi:** gövdede tahta kaplama sıraları, kopmuş halatlar, cıvadra, pencereli kıç
  köşkü ve fener, karinada midye ve yosun.
- **Deniz feneri:** taş derzleri, çerçeveli pencere ve kapı, kapı basamakları, balkon
  konsolları, lamba ve pirinç halkalar, rüzgâr gülü.
- Saz çatılar katmanlı, kenarları tırtıklı, tepesi bağlı; meşaleler bez sargılı.
- **Zemin:** yumuşak gölgeli, ince tümsekli arazi; çimende ton lekeleri, kuru yamalar ve
  kumlu kenarlar, kumda dalga izleri, ıslak kum şeridi ve köpük çizgisi, kayalık yamaçlarda
  tabakalar, patikaların ortası daha aşınmış. Dik kaya yüzleri köşeli kalır.
- **Doğal öğeler:** kayalar iki ölçekte engebeli, tabakalı, yüz yüz üç tonlu, üstleri
  yosunlu; çalılar yaprak yaprak, bir kısmı beş taçyapraklı çiçekli; çim öbekleri kıvrık,
  dibi koyu ucu açık; uzun çim, palmiye altlarında eğrelti otları, sığ suda deniz çayırı,
  kıyıda dalgaya vuran kayalar, patika kenarlarında taşlar, kumsalda kumul otu, kabuk, deniz
  yıldızı ve çakıl.
- **Eşyalar:** sandıklar tahta tahta, perçinli, köşebentli, menteşeli ve kilitli; fıçılar
  çıtalı ve çember kuşaklı; kasalar tahta kaplamalı, çerçeveli ve çaprazlı.

## Hazine noktaları

Her `TreasureSpawn_##` boş nesnesinin altında bir `TreasureChest_##` sandığı vardır. Oyun
her turda birini (ya da birkaçını) açık bırakıp gerisini gizleyebilir.

| # | Yer | Not |
| --- | --- | --- |
| 1 | Tapınak sunağı | Açık kapaklı sandık, altın yığını |
| 2 | Batık gemi güvertesi | Yarığın yanında |
| 3 | Kaya kemerinin altı | Yanında birkaç altın |
| 4 | Korsan kampı çadırı | Çadırın içinde |
| 5 | Adacık | Palmiyelerin altında |
| 6 | Güneydoğu kumsalı | Kırmızı **X** işaretinin üstünde, yarı gömülü; yanında kürek |
| 7 | Tapınağın arkası | Devrik sütunların arasında |
| 8 | Kuzey kumsalı | Büyük kayaların arasında |
| 9 | Deniz feneri balkonu | Spiral merdivenle çıkılır |
| 10 | Balıkçı kulübesi | Kulübenin içinde, arka köşede |
| 11 | Yıkık ev | Duvarların içinde |
| 12 | Mezarlık | Açık mezarın yanında |
| 13 | Gözcü kulesi | Platformun üstünde |
| 14 | Kaçakçı barakası | Fıçıların önünde |

`PlayerSpawn_01..06` iskelenin kumsal ucunda, iki sıra hâlinde, yüzleri adaya dönük.

## Hazine olabilecek 200 eşya

Adaya, her biri ayrı bir nesne olan 200 eşya (`Prop_001_<çeşit>` … `Prop_200_<çeşit>`)
dağıtıldı; oyun hazineyi bunlardan herhangi birinin içine/altına saklayabilir. 38 çeşit var
ve her bölgeye oraya yakışan eşyalar düşer:

![catalog](renders/props_catalog.png)

| Bölge | Adet | Eşyalar |
| --- | --- | --- |
| Kumsal | 36 | deniz kabuğu, deniz yıldızı, şişede mektup, hindistan cevizi, yengeç, dalgaya vurmuş kütük, kova, halat, küçük sandık, korsan bayrağı |
| Palmiye altları | 19 | hindistan cevizi (tek / yığın), çuval, kova |
| Tapınak | 27 | amfora, taş küp, kil çömlek, altın başlı put, kafatası, mücevher kümesi, mumlar, kitap, parşömen, çan, miğfer, kılıç, kese |
| Korsan kampı | 17 | fıçı, sandık kasa, çuval, fener, kova, rom şişesi, kazan, halat, kese, parşömen, kitap, küçük sandık, kürek, maşrapa |
| Batık gemi çevresi | 18 | gülle yığını, çapa, dümen, top, fıçı, kasa, rom şişesi, şişe, miğfer, kılıç, halat, çan, fener |
| Kaya kemeri | 10 | kafatası, fener, mücevher, mantarlar, mumlar, çömlek, kılıç, kese |
| İskele | 6 | kasa, fıçı, halat, kova, fener, çuval |
| Adacık | 5 | hindistan cevizi, kabuk, şişe, deniz yıldızı |
| Deniz feneri | 7 | fener, halat, kova, kasa, fıçı, şişe, kitap, maşrapa |
| Balıkçı kulübesi | 8 | kova, halat, çuval, fıçı, kasa, hindistan cevizi, şişe, fener, maşrapa |
| Yıkık ev | 8 | çömlek, taş küp, kitap, mumlar, kafatası, kese, kazan, parşömen, küçük sandık |
| Mezarlık | 6 | kafatası, mumlar, kürek, fener, kılıç, miğfer, kese, mantarlar |
| Gözcü kulesi | 4 | fıçı, kasa, fener, çuval, gülle |
| Kuyu | 2 | kova, kese |
| Kaçakçı barakası | 5 | fıçı, kasa, rom şişesi, çuval, halat, küçük sandık |
| İç kesim | 22 | mantarlar, kütük, çuval, çömlek, kürek, kılıç, miğfer, fener, kafatası, mücevher, küçük sandık, bayrak, amfora… |

Eşyalar yapıların, ağaçların, kayaların, patikaların ve spawn noktalarının içine girmeyecek
şekilde yerleştirilir; tapınaktakiler platform katlarının üstüne oturur. Aynı çeşitteki
eşyalar tek bir mesh'i paylaşır (glb'de her şekil bir kez saklanır).

## Motor notları

- glb toplam ~575 bin üçgen; bunun büyük kısmı zemin örtüsü (çim, çiçek, çakıl) ve bitki
  örtüsü. Düşük donanım için `Grass`, `Flowers`, `Tall_Grass` gibi zemin örtüsü nesneleri
  mesafeye göre gizlenebilir ya da kapatılabilir.
- Birim metre, glTF'de +Y yukarı. Kuzey Blender'da +Y, glTF'de -Z.
- Boş nesnelerin glTF `extras` alanında `kind` (`treasure_spawn`, `player_spawn`),
  `index` ve hazine noktaları için `label` bulunur; sandıklarda `kind = treasure_chest`.
- Eşyalarda `kind = treasure_prop`, `item` (çeşit, ör. `barrel`, `coconut`), `zone` (bölge)
  ve `index` (1–200) vardır.
- `Terrain` renklerini köşe rengi (`COLOR_0`) olarak taşır; çarpışma için doğrudan
  kullanılabilir. `Water` yarı saydam bir düzlemdir (z = 0), `Seabed` ufka kadar uzanan
  derin deniz zeminidir; ikisinin de çarpışması kapatılmalı.
- Adanın dışı derin su (-3,5 m); oyuncuları tutmak için yaklaşık 36 m yarıçaplı görünmez
  bir sınır yeterli.
- Meşale ve kamp ateşi alevleri (`M_Flame`, `M_FlameCore`) emissive; bloom açıkken en iyi görünür.

## Yeniden üretmek

```
blender --background --python build_treasure_cove.py
blender --background --python build_treasure_cove.py -- --fast
blender --background --python build_treasure_cove.py -- --no-render
```

Parşömen haritası için Pillow gerekir; yoksa o adım atlanır.
