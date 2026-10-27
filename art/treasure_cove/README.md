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
| `renders/cove_*.png` | Önizlemeler: genel görünüm, tapınak, tepeden |

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
- Saklanmak için iki düzine kadar palmiye, ~25 çalı ve ~25 kaya; kuzey kumsalında arkasına saklanılabilecek
  büyük kayalar.

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

`PlayerSpawn_01..06` iskelenin kumsal ucunda, iki sıra hâlinde, yüzleri adaya dönük.

## Motor notları

- Birim metre, glTF'de +Y yukarı. Kuzey Blender'da +Y, glTF'de -Z.
- Boş nesnelerin glTF `extras` alanında `kind` (`treasure_spawn`, `player_spawn`),
  `index` ve hazine noktaları için `label` bulunur; sandıklarda `kind = treasure_chest`.
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
