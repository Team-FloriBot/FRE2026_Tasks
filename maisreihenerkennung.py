import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import time

# ==========================================
# EINSTELLUNGEN FÜR DIE SIMULATION
# ==========================================
# Wähle das Fahrmuster des Roboters:
# "jede_gasse"         : Fährt in jede benachbarte Gasse (fügt 1 neue Reihe pro Schritt hinzu)
# "jede_zweite_gasse"  : Überspringt eine Gasse (fügt 2 neue Reihen pro Schritt hinzu)
FAHRMUSTER = "jede_zweite_gasse" 


class Row:
    """Repräsentiert eine einzelne getrackte Pflanzenreihe über mehrere Frames hinweg."""
    def __init__(self, start_x, direction, row_id):
        self.id = row_id
        self.direction = direction # 1 = wächst von unten nach oben, -1 = von oben nach unten
        self.last_x = start_x      # X-Position am Start (für Abstandsvergleiche)
        self.centers = ([], [])    # Getrackte Mittelpunkte der Pflanzengruppen
        self.spline_data = None    # Berechnete Spline-Kurve (Weg der Reihe)
        self.missed_frames = 0     # Zähler für Frames, in denen die Reihe nicht gesehen wurde
        self.active = True         # Ist die Reihe aktuell noch im Bild?

    def update(self, grid):
        """Aktualisiert die Reihenposition basierend auf dem aktuellen Bild-Grid."""
        height, width = grid.shape
        window_size = 30
        step_size = 15
        current_x = self.last_x
        res_x, res_y = [], []
        empty_windows = 0
        found_any = False
        
        # Suchrichtung festlegen
        if self.direction == 1:
            y_range = range(0, height - window_size, step_size)
        else:
            y_range = range(height - window_size, 0, -step_size)

        # Entlang der Y-Achse nach Pflanzen-Clustern suchen
        for y in y_range:
            # Begrenzter Suchbereich in X-Richtung, um nicht auf Nachbarreihen zu springen
            x_min = max(0, int(current_x - 40))
            x_max = min(width, int(current_x + 40))
            window = grid[y:y+window_size, x_min:x_max]
            points = np.argwhere(window > 0)
            
            # Wenn genug "Pflanzenpixel" gefunden wurden
            if len(points) >= 5:
                pts_global = points + [y, x_min]
                mean_y, mean_x = np.mean(pts_global, axis=0) # Mittelpunkt des Clusters
                res_x.append(mean_x)
                res_y.append(mean_y)
                current_x = mean_x
                empty_windows = 0
                found_any = True
            else:
                # Abbruchbedingung: Wenn wir die Reihe verlieren (Lücke zu groß)
                if found_any:
                    empty_windows += 1
                    if empty_windows >= 2: break
        
        # Wenn wir eine gültige Reihe getrackt haben (min. 4 Punkte)
        if len(res_x) >= 4:
            self.centers = (res_x, res_y)
            self.last_x = np.mean(res_x) # Durchschnittliches X für stabile Abstandsvergleiche
            self.missed_frames = 0
            
            # Spline-Interpolation für einen glatten Kurvenverlauf
            sy, sx = np.array(res_y), np.array(res_x)
            sort_idx = np.argsort(sy)
            sy, sx = sy[sort_idx], sx[sort_idx]
            try:
                k = min(3, len(sy) - 1)
                spline = UnivariateSpline(sy, sx, s=60, k=k)
                y_fine = np.linspace(sy.min(), sy.max(), 50)
                self.spline_data = (y_fine, spline(y_fine))
            except:
                self.spline_data = None
        else:
            # Gedächtnis: Reihe wird nicht sofort gelöscht, sondern darf kurz verschwinden
            self.missed_frames += 1
            if self.missed_frames > 5:
                self.active = False

class CornFieldTracker:
    """Verwaltet alle erkannten Pflanzenreihen und generiert den Fahrpfad."""
    def __init__(self, min_row_dist=50):
        self.rows = []
        self.min_row_dist = min_row_dist # Mindestabstand zwischen zwei Reihen
        self.next_id = 1

    def process_frame(self, grid):
        """Hauptfunktion, die in jedem Frame/Simulationsschritt aufgerufen wird."""
        # 1. Bestehende Reihen aktualisieren
        for row in self.rows:
            if row.active: row.update(grid)
            
        # Tote Reihen aussortieren
        self.rows = [r for r in self.rows if r.active]
        
        # 2. Nach neuen Reihen am Bildrand suchen
        self._detect_new_rows(grid)
        
        # 3. Doppelte Erkennungen auf derselben Reihe bereinigen
        self._remove_duplicates()
        
        # 4. Reihen strikt von links nach rechts sortieren (wichtig für Gassen-Logik)
        self.rows.sort(key=lambda r: r.last_x)

    def _remove_duplicates(self):
        """Sorgt dafür, dass keine zwei getrackten Reihen auf der gleichen echten Pflanze liegen."""
        to_remove = set()
        for i in range(len(self.rows)):
            if i in to_remove: continue
            for j in range(i + 1, len(self.rows)):
                if j in to_remove: continue
                r1, r2 = self.rows[i], self.rows[j]
                
                # Wenn Reihen zu nah beieinander liegen, lösche die schwächere
                if abs(r1.last_x - r2.last_x) < self.min_row_dist:
                    if len(r1.centers[0]) >= len(r2.centers[0]):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
        
        self.rows = [r for i, r in enumerate(self.rows) if i not in to_remove]

    def _detect_new_rows(self, grid):
        """Sucht mithilfe eines Histogramms am oberen und unteren Bildrand nach neuen Reihenanfängen."""
        height, width = grid.shape
        hist_bottom = np.sum(grid[0:300, :], axis=0)
        hist_top = np.sum(grid[height-300:height, :], axis=0)
        
        def get_peaks(hist):
            thresh = np.mean(hist) * 1.8
            peaks = []
            for x in range(50, width-50):
                if hist[x] > thresh and hist[x] == np.max(hist[x-25:x+26]):
                    peaks.append(x)
            return peaks
            
        # Neue Reihen von unten
        for p in get_peaks(hist_bottom):
            if not any(abs(r.last_x - p) < self.min_row_dist for r in self.rows):
                new_row = Row(p, 1, self.next_id)
                self.next_id += 1
                new_row.update(grid)
                if new_row.active: self.rows.append(new_row)
                
        # Neue Reihen von oben
        for p in get_peaks(hist_top):
            if not any(abs(r.last_x - p) < self.min_row_dist for r in self.rows):
                new_row = Row(p, -1, self.next_id)
                self.next_id += 1
                new_row.update(grid)
                if new_row.active: self.rows.append(new_row)

    def get_planning_path(self):
        """Berechnet den Fahrpfad (blau) in der Mitte der Gassen."""
        if len(self.rows) < 2: return None
        full_path_x, full_path_y = [], []
        
        path_idx = 0
        for i in range(len(self.rows) - 1):
            r1, r2 = self.rows[i], self.rows[i+1]
            if not (r1.spline_data and r2.spline_data): continue
            
            # Gasse nur zeichnen, wenn der Abstand zwischen den Reihen realistisch ist
            if abs(r1.last_x - r2.last_x) < self.min_row_dist: continue
            
            y1, x1 = r1.spline_data
            y2, x2 = r2.spline_data
            y_min = max(y1.min(), y2.min())
            y_max = min(y1.max(), y2.max())
            if y_max <= y_min: continue
            
            # Pfad exakt in die Mitte zwischen den beiden Splines legen
            y_mid = np.linspace(y_min, y_max, 20)
            x_mid = (np.interp(y_mid, y1, x1) + np.interp(y_mid, y2, x2)) / 2
            
            # Bei "jede_zweite_gasse" wird nur in jeder ZWEITEN Lücke ein Pfad gezeichnet
            if FAHRMUSTER == "jede_zweite_gasse" and i % 2 == 1:
                continue

            # Schlangenlinien: Abwechselnd von unten nach oben und oben nach unten fahren
            if path_idx % 2 == 1: 
                y_mid, x_mid = y_mid[::-1], x_mid[::-1]
            
            # Wendemanöver berechnen (Verbindung zwischen zwei Gassen am Feldende)
            if full_path_x:
                prev_x, prev_y = full_path_x[-1], full_path_y[-1]
                next_x, next_y = x_mid[0], y_mid[0]
                
                # Der Bogen wird nach außen gezogen (Vorgewende), um nicht über Pflanzen zu fahren
                offset = 80 if prev_y > 800 else -80
                cx = np.linspace(prev_x, next_x, 20)
                cy = np.linspace(prev_y, next_y, 20) + offset * np.sin(np.linspace(0, np.pi, 20))
                full_path_x.extend(cx.tolist()); full_path_y.extend(cy.tolist())

            full_path_x.extend(x_mid.tolist())
            full_path_y.extend(y_mid.tolist())
            path_idx += 1
            
        return full_path_x, full_path_y

def create_corn_field(rows_to_create, start, percent, width=1600, height=1600, row_spacing=75, rows_total=14):
    """Generiert künstliche Maisreihen mit etwas organischem 'Schwung'."""
    grid = np.zeros((height, width))
    y_coords = np.arange(height)
    row_bases = list(range(row_spacing, row_spacing * (rows_total + 1), row_spacing))
    for i in rows_to_create:
        if i >= len(row_bases): continue
        row_x = row_bases[i] + 20 * np.sin(y_coords / 50)
        for y, x in zip(y_coords, row_x):
            if start == "unten":
                if y % 30 == 0 and y > 200 and y < height - 200 - (100-percent)*(height-400)/100:
                    grid[y-2:y+3, int(x)-2:int(x)+3] = 1
            else:
                if y % 30 == 0 and y > 200 + (100-percent)*(height-400)/100 and y < height - 200:
                    grid[y-2:y+3, int(x)-2:int(x)+3] = 1
    return grid

# ==========================================
# HAUPTPROGRAMM / SIMULATIONSSCHLEIFE
# ==========================================
plt.ion()
fig = plt.figure(figsize=(8, 10))
tracker = CornFieldTracker(min_row_dist=50)
lines_total = 14
grid = np.zeros((1600, 1600))
# Etwas zufälliges Bildrauschen hinzufügen
grid += np.random.rand(1600, 1600) > 0.9995

# Je nach Fahrmuster ändert sich die Anzahl der Durchgänge
steps = lines_total - 1 if FAHRMUSTER == "jede_gasse" else (lines_total // 2)

for step_idx in range(steps):
    # Richtung wechselt nach jeder Durchfahrt
    side = "unten" if step_idx % 2 == 0 else "oben"
    
    # Berechne, welche Reihen in diesem Durchgang ins Sichtfeld kommen
    if step_idx == 0:
        rows_to_add = [0, 1]
        step_text = "Initiale Einfahrt (Reihe 1 & 2)"
    else:
        if FAHRMUSTER == "jede_gasse":
            rows_to_add = [step_idx + 1]
            step_text = f"Wende zu Reihe {step_idx + 2}"
        else: # jede_zweite_gasse
            rows_to_add = [step_idx * 2, step_idx * 2 + 1]
            step_text = f"Wende (Überspringe Gasse) zu Reihen {step_idx*2+1} & {step_idx*2+2}"

    # Simuliere das "Einfahren" ins Feld durch schrittweises Einblenden (0% bis 100%)
    for percent in [0, 2, 4, 6, 8, 10, 20, 30, 50, 70, 80, 90, 92, 94, 96, 98, 100]:
        # Erstelle neues Pflanzenmaterial für die aktuellen Reihen
        grid = grid + create_corn_field(rows_to_add, side, percent, rows_total=lines_total+1)
        
        # Tracker auf das Bild anwenden
        tracker.process_frame(grid)
        
        # Anzeige zurücksetzen und Grid zeichnen
        plt.clf()
        plt.imshow(grid > 0, cmap='gray_r', origin='lower')
        
        
        # 1. Erkannte Pflanzenreihen (Grün) zeichnen
        for r in tracker.rows:
            if r.spline_data:
                plt.plot(r.spline_data[1], r.spline_data[0], color='green', linewidth=1.5, alpha=0.6)
        
        # 2. Geplanten Weg (Blau) zeichnen
        path = tracker.get_planning_path()
        if path:
            plt.plot(path[0], path[1], color='#0077FF', linewidth=2.5, label="Navigationspfad", zorder=10)
        
            
        # 3. HUD (Info-Text) einblenden
        info_text = f"Modus: {FAHRMUSTER}\n{step_text}\nRichtung: {side}\nSichtbarkeit: {percent}%\nErkannte Reihen: {len(tracker.rows)}"
        plt.text(50, 1450, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        plt.xlim(0, 1600); plt.ylim(0, 1600)
        plt.legend(loc='upper right')
        plt.draw()
        plt.pause(0.5)

plt.ioff()
plt.show()
