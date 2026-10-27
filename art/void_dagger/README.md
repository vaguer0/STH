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

- **Namlu:** hafif S kıvrımlı koyu bir çekirdek; kenarları düzensiz, grotesk dişlere
  bölünür. Her diş farklı bir kalıptan gelir (sivri diş, geriye kıvrılan kanca, çatal,
  kırık/yontuk diş, testere kümesi, uzun ince diken), boyu, eğimi (bazıları geriye bakar) ve
  dip derinliği rastgeledir; kontura ayrıca el titremesi gibi hafif bir dalgalanma eklenir.
  Keskin ağız tarafı büyük dişli, sırt seyrek ve küçük dişli. Kontur kendini kesmeyene kadar
  tohum denenir, kısıtlı Delaunay ile üçgenlenir; kesit düz plaka, yalnızca kenarda bileme.
- **Doku:** mürekkep çizgisiyle çevrili koyu çekirdek, uca doğru mordan lilaya geçen dişler,
  açık bileme şeridi; omurga boyunca parlayan bir damar ve bazı dişlere uzanan kolları.
- **Kabza:** namlu dibini saran topaklı, organik pullu bir düğüm; düğümden her yöne (yassı
  yüzlere doğru olanlar dahil) çıkan kalın, boğumlu, kıvrık boynuzlar: keskin ağız tarafında
  namlu boyunca yükselen büyük bir boynuz, sapa doğru kıvrılanlar, yanlara fırlayanlar.
  Boynuzlar kökte koyu, halka halka boğumlu, uca doğru parlayan lilaya döner. Sap ejderha
  kuyruğu gibi pullu halkalar; topuzda küçük boğumlu dikenler.
- **Gölge alevi:** diş uçlarından (hepsinden değil) yukarı kıvrılan üç tonlu el çizimi
  alevler, uçta küçük bir alev, yükselip sönen kıvılcımlar. 6 çizim üçerli değişir
  (0,75 sn döngü); damar ve boynuz uçları yavaş nabızla parlar.

## Motor notları

Orijin sapın ortasında, glTF'de hançer +Y yönünde. `FX_ShadowFlame_*` ve
`Dagger_InkOutline` için gölge düşürmeyi kapatın; bloom açık olursa en iyi görünür.

## Yeniden üretmek

```
blender --background --python build_void_dagger.py
blender --background --python build_void_dagger.py -- --fast
```
