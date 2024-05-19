# Piano Tiles computer vision

# **Objectif du projet**

L’objectif est d’automatiser notre jeu Piano Tiles par du traitement d’images pour atteindre des scores beaucoup plus élevés. Nous chercherons d’abord à capturer l’écran de notre jeu pour isoler la couleur noire, puis nous détecterons la présence ou non de tuiles. Enfin, nous automatiserons la gestion du clavier pour cliquer dessus.

Pour notre projet, nous aurons besoin des bibliothèques suivantes :

- **`NumPy`** : pour la manipulation des tableaux de pixels de nos images
- **`Pillow`** : pour capturer l’image de notre écran
- **`Pynput`**/**`keyboard`** : pour le contrôle du clavier et de la souris
- **`OpenCV`** : pour le traitement d’images

# I- Capture de notre écran

Dans un premier temps, pour pouvoir automatiser notre jeu Piano Tiles, nous devons pouvoir récupérer l’image de notre écran en continu.

Pour cela, à travers une boucle infinie, nous effectuons une capture d’écran avec la bibliothèque Pillow : `img = ImageGrab.grab()`. Nous obtenons alors des images au format `PIL.Image`.

Nous convertissons ensuite l'image PIL en un tableau NumPy pour la traiter facilement avec OpenCV : `img_np = np.array(img)`. Nous obtenons alors un tableau qui contient la valeur des pixels de notre capture d’écran. Dans mon cas, la dimension de l’écran est de 1920x1080.

## II- Application du masque de couleurs

Dans un second temps, nous souhaitons traiter l’image de notre capture d’écran afin d’isoler la couleur noire correspondant aux tuiles sur lesquelles nous souhaitons cliquer.

Nous commençons par réduire de moitié la taille de la capture d’écran, puis nous passons de l’espace colorimétrique RGB à HSV, couramment utilisé pour la segmentation des couleurs.

```python
frame = cv2.resize(img_np, (int(img_np.shape[1] * 0.5), int(img_np.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)
frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```

Nous appliquons ensuite un effet de flou sur notre image pour lisser les détails et améliorer la détection des contours des tuiles.

```python
frameBlur = cv2.blur(frameHSV, (5, 5))
```

Enfin, nous utilisons la méthode `inRange()` pour créer un masque de couleur.

```python
frameMask = cv2.inRange(frameBlur, LOWER, HIGHER)
```

`LOWER` et `HIGHER` sont deux tuples qui nous permettent d'isoler une certaine couleur dans notre image, dans ce cas le noir. Nous devons spécifier les plages de teinte, de saturation et de valeur de cette couleur. `LOWER` et `HIGHER` définissent les limites inférieure et supérieure de la plage de couleurs que nous cherchons à détecter.

En effectuant cela, nous créons un masque binaire où les pixels de l'image HSV qui se trouvent dans la plage de couleurs que nous avons définie sont mis à 255 (apparaissent donc en blanc), tandis que tous les autres pixels sont mis à 0 (apparaissent donc en noir).

Pour la détection du noir, nous aurons :

- `LOWER = (0, 0, 0)`
- `HIGHER = (180, 255, 30)`
  
![Screenshot (867)](https://github.com/Haki-i/PianoTiles-computerVision/assets/137703849/fe54dd14-96cf-412c-bf7b-f554556a31ff)

# III- Détection des tuiles

Une fois la couleur noire isolée, nous allons déterminer tous les contours trouvés sur notre capture de jeu.

```python
elements = cv2.findContours(frameMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
```

Parmi tous les contours détectés dans l'image, nous sélectionnons celui ayant la plus grande aire. Ensuite, nous recherchons le plus petit rectangle capable d'englober ce contour. Lorsqu’une tuile noire apparaîtra à l’écran, notre programme pourra l’encadrer et récupérer les coordonnées de son coin haut gauche ainsi que sa longueur et largeur.

```python

if len(elements) > 0:
    c = max(elements, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
```

Avec ces informations, nous calculons l’aire de la tuile à l’écran et nous déterminons un seuil limite indiquant l’aire nécessaire à dépasser pour que l’on considère que le contour détecté est bien celui d’une tuile et non d’un élément parasite.

Remarque :

- Plus l’aire du seuil est élevée, plus la tuile devra apparaître dans son entièreté à l’écran pour que le programme soit capable de la détecter.
- Plus l’aire du seuil est faible, plus le programme sera en mesure de détecter la tuile avant même qu’elle ne soit complètement affichée. Cela permettra de cliquer plus rapidement sur la tuile.

Il faut donc trouver un juste milieu pour avoir un seuil bas sans toutefois détecter autre chose qu’une tuile.

Dans le cas où une tuile est détectée, nous calculons son centre.

```python
area = w * h
if area > 100:  # Aire suffisante pour une tuile détectée
        # Centre de la tuile par rapport à frame
        cx = x + w / 2
        cy = y + h / 2
```

Si notre fonction de détection de contours a détecté une tuile, alors la valeur renvoyée sera les coordonnées de son centre ainsi que sa hauteur. Sinon, nous renvoyons `None`.

# IV- Appui sur la tuile

La dernière étape consiste à cliquer sur la tuile une fois que celle-ci est détectée. Le contrôle automatique du clavier ne se fera uniquement que lorsque nous appuierons sur la touche entrée. De même, lorsque nous souhaiterons l'arrêter.

```python
  if keyboard.is_pressed("return"):
            begin = not begin  # On active ou non le clavier
```

Si notre objet `tile` n’est pas `None`, cela signifie qu’une tuile est détectée. À ce moment-là, nous pouvons récupérer les informations de celle-ci.

Pour déplacer notre souris au bon endroit sur notre écran, nous devons faire correspondre les coordonnées du centre de la tuile sur la capture d’image par rapport aux véritables dimensions de notre jeu.

Remarque : la dimension de mon écran est 1920x1080, cependant en utilisant `pynput`, on observe que sa dimension se limite à 1535x863. Nous utiliserons donc ces dimensions pour la conversion.

Enfin, dans l’objectif d’optimiser l’efficacité de notre programme, nous ne cliquerons pas directement au centre de la tuile mais le plus bas possible. En effet, lorsque le jeu s'accélérera, les tuiles descendront beaucoup plus rapidement. Le temps que la souris se déplace, la tuile aura changé de place et nous ne cliquerons plus au niveau du centre mais plus haut. Pour pallier cet écart, nous anticiperons en cliquant le plus bas possible.

Chaque fois qu’une tuile est pressée, sa couleur change et notre programme passe automatiquement à la suivante.

```python
tile = detect_tile(frameMask)

if begin and tile:  # Si clavier activé et tile n'est pas None (donc détection)

            cx, cy, h = tile

            # On calcule les coordonnées du centre de la tuile par rapport à notre écran
            cx_screen = (cx * SCREEN_WIDTH) / FRAME_WIDTH
            cy_screen = (cy * SCREEN_HEIGHT) / FRAME_HEIGHT

            # On met la souris plus bas sur la tuile pour optimiser
            cy_screen_lower = cy_screen + 0.9*h

            mouse.position = (cx_screen, cy_screen_lower)
            mouse.click(Button.left, 1)
```

![20240518_224200-ezgif com-optimize](https://github.com/Haki-i/PianoTiles-computerVision/assets/137703849/3778885e-eadb-49fc-a352-355a14077443)
