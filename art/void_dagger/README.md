# Boşluk Hançeri — SeekThenHide

Mor hançerin, verilen referans hançerlerin (Kamish's Wrath, Baruka, Demon King,
Knight Killer) yapısı ve stiliyle yeniden tasarlanmış hali. Katanalarla aynı araç
setini (`../mystic_katana/build_mystic_katana.py`) kullanır; her şey
`build_void_dagger.py` ile Blender içinde üretilir.

![hero](renders/dagger_hero.png)

## Dosyalar

| Dosya | Ne işe yarar |
| --- | --- |
| `export/void_dagger.glb` | Oyuna atılacak dosya (dokular gömülü, `ShadowFlameLoop` animasyonu) |
| `void_dagger.blend` | Düzenlenebilir sahne; dokular dosyanın içine paketli |
| `textures/` | Namlu (iki yüz yan yana, 1024×2048) ve kabza atlası (1024²) |
| `renders/` | Önizlemeler, `dagger_shadow_loop.mp4` alev döngüsü videosu |

## Blender'da oynatma

`void_dagger.blend`'i açın, 3D görünümde **Z → Material Preview** (ya da bloom ile
**Rendered**) seçin, **Space**'e basın. 1–72 kare (24 fps) döngü.

## Tasarım

Kamish's Wrath'ın yapısı, mevcut mor hançerin renkleriyle.

- **Namlu:** dikensiz, geniş, hafif S kıvrımlı bir hançer namlusu; ama kenarları kusursuz
  bir CAD eğrisi değil: yavaş, elde dövülmüş gibi bir dalgalanma, keskin ağızda altı, sırtta
  üç rastgele çentik/yontuk ve konturun tamamında hafif el titremesi. Kısıtlı Delaunay ile
  üçgenlenir; kesit düz plaka, yalnızca kenarda bileme.
- **Doku:** ortada mürekkep çizgisiyle çevrili koyu oluk bandı, kenarlara doğru mordan lilaya
  geçiş, açık bileme şeridi; omurga boyunca parlayan bir damar ve kenarlara uzanan kolları.
- **Kabza:** yassı, dövme metal "yarasa kanadı" çapraz muhafız: iki yana açılan kalın
  kanatların üst kenarı namluya doğru kıvrılıp pençeyle biter, alt kenarı kanat parmakları
  gibi dalgalı; ortada elmas gövde. Gövdede ve kanatlarda kabartma, parlayan mor mine
  kakmalar; her parçanın altında kalın mürekkep kontur katmanı. Keskin ağız tarafındaki kanat
  biraz daha büyük. Sap ejderha kuyruğu gibi pullu halkalar, sivri topuz.
- **Gölge alevi:** keskin ağızdan (ve sırttan daha seyrek) yukarı kıvrılan üç tonlu el
  çizimi alevler, uçta küçük bir alev, yükselip sönen kıvılcımlar. 6 çizim üçerli değişir
  (0,75 sn döngü); damar ve kakmalar yavaş nabızla parlar.

## Motor notları

Orijin sapın ortasında, glTF'de hançer +Y yönünde. `FX_ShadowFlame_*` ve
`Dagger_InkOutline` için gölge düşürmeyi kapatın; bloom açık olursa en iyi görünür.

## Yeniden üretmek

```
blender --background --python build_void_dagger.py
blender --background --python build_void_dagger.py -- --fast
```
