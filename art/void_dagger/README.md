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

- **Namlu:** hafif S kıvrımlı koyu bir çekirdek; keskin ağız tarafı düzensiz aralıklı,
  farklı boyutlarda, uca doğru eğik büyük alev dişlerine bölünür, sırt tarafı sakin ve
  seyrek küçük dişli; kıvrık sivri uç. Kontur prosedürel olarak üretilir (kendini kesmeyene
  kadar tohum denenir), kısıtlı Delaunay ile üçgenlenir; kesit düz plaka, yalnızca kenarda
  dar bileme.
- **Doku:** mürekkep çizgisiyle çevrili koyu çekirdek, uca doğru mordan lilaya geçen dişler,
  açık bileme şeridi; omurga boyunca parlayan bir damar ve bazı dişlere uzanan kolları.
- **Kabza:** pullu, kalkan biçimli bir ejderha kafası ve iki yüzünde parlayan badem göz;
  kafadan namlunun dişlerini sürdüren kalın, katmanlı alev dilleri (her yanda üçü yukarı,
  ikisi sapa doğru; keskin ağız tarafında daha büyük). Sap ejderha kuyruğu gibi üst üste
  binen pullu halkalardan; topuzda aşağı yalayan alev dilleri.
- **Gölge alevi:** diş uçlarından (hepsinden değil) yukarı kıvrılan üç tonlu el çizimi
  alevler, uçta küçük bir alev, yükselip sönen kıvılcımlar. 6 çizim üçerli değişir
  (0,75 sn döngü); damar ve göz yavaş nabızla parlar.

## Motor notları

Orijin sapın ortasında, glTF'de hançer +Y yönünde. `FX_ShadowFlame_*` ve
`Dagger_InkOutline` için gölge düşürmeyi kapatın; bloom açık olursa en iyi görünür.

## Yeniden üretmek

```
blender --background --python build_void_dagger.py
blender --background --python build_void_dagger.py -- --fast
```
