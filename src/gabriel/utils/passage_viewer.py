try:  # pragma: no cover - optional dependency
    import tkinter as tk
    from tkinter import ttk, scrolledtext
except Exception:  # pragma: no cover - optional dependency
    tk = None  # type: ignore
    ttk = None  # type: ignore
    scrolledtext = None  # type: ignore

import html
import pandas as pd
import random
import re
from typing import Iterable, List, Dict, Any, Optional, Sequence, Tuple, Union
import colorsys
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def _generate_distinct_colors(n: int) -> List[str]:
    """Generate ``n`` visually distinct hex colors.

    This helper is shared by both the rich ``tkinter`` viewer and the simpler
    HTML based viewer used in headless environments such as Google Colab.
    """

    base_colors: List[str] = []
    if plt is not None:
        if n <= 20:
            cmap = plt.get_cmap("tab20")
            for i in range(n):
                rgb = cmap(i)[:3]
                base_colors.append(
                    "#{:02x}{:02x}{:02x}".format(
                        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                    )
                )
            return base_colors
        else:
            cmap = plt.get_cmap("tab20")
            for i in range(20):
                rgb = cmap(i)[:3]
                base_colors.append(
                    "#{:02x}{:02x}{:02x}".format(
                        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                    )
                )

    for i in range(len(base_colors), n):
        hue = (i * 1.0 / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
        base_colors.append(
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        )
    return base_colors[:n]

class PassageViewer:
    def __init__(self, df: pd.DataFrame, column_name: str, categories: Optional[Union[List[str], str]] = None):
        self.df = df.copy()
        self.column_name = column_name
        self.current_index = 0
        self.last_tooltip_cats = None
        self.selected_snippet_tag = None
        self.dark_mode = True  # Default to dark mode
        # Detect mode: static categories or dynamic coded_passages
        if categories is None and 'coded_passages' in df.columns:
            self.dynamic_mode = True
            all_categories = set()
            for coded_passages in df['coded_passages']:
                if coded_passages and isinstance(coded_passages, dict):
                    all_categories.update(coded_passages.keys())
            self.categories = sorted(list(all_categories))
        elif isinstance(categories, str) and categories == 'coded_passages':
            self.dynamic_mode = True
            all_categories = set()
            for coded_passages in df['coded_passages']:
                if coded_passages and isinstance(coded_passages, dict):
                    all_categories.update(coded_passages.keys())
            self.categories = sorted(list(all_categories))
        else:
            self.dynamic_mode = False
            self.categories = categories if categories else []
        self.colors = _generate_distinct_colors(len(self.categories))
        self.category_colors = dict(zip(self.categories, self.colors))
        self.tooltip = None
        self._setup_gui()
        self._display_current_text()

    def _setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Passage Viewer - Modern Text Analysis")
        self.root.geometry("1600x1000")
        self._apply_theme()

    def _apply_theme(self):
        # Set up theme colors and fonts
        if self.dark_mode:
            bg_main = '#181a1b'
            bg_secondary = '#23272a'
            text_primary = '#f7f7f7'
            text_accent = '#00bcd4'
            text_info = '#b0b0b0'
            legend_border = '#444'
            highlight_sel = '#fff176'
            font_main = ('Quicksand', 20, 'bold')
            font_header = ('Quicksand', 16, 'bold')
            font_legend = ('Quicksand', 15)
            font_info = ('Quicksand', 14)
            font_text = ('Quicksand', 20)
            font_popup = ('Quicksand', 22, 'bold')
        else:
            bg_main = '#f7f7f7'
            bg_secondary = '#eaeaea'
            text_primary = '#23272a'
            text_accent = '#00bcd4'
            text_info = '#444'
            legend_border = '#bbb'
            highlight_sel = '#fff176'
            font_main = ('Quicksand', 20, 'bold')
            font_header = ('Quicksand', 16, 'bold')
            font_legend = ('Quicksand', 15)
            font_info = ('Quicksand', 14)
            font_text = ('Quicksand', 20)
            font_popup = ('Quicksand', 22, 'bold')
        self.bg_main = bg_main
        self.bg_secondary = bg_secondary
        self.text_primary = text_primary
        self.text_accent = text_accent
        self.text_info = text_info
        self.legend_border = legend_border
        self.highlight_sel = highlight_sel
        self.font_main = font_main
        self.font_header = font_header
        self.font_legend = font_legend
        self.font_info = font_info
        self.font_text = font_text
        self.font_popup = font_popup
        self._build_gui()

    def _build_gui(self):
        for widget in self.root.winfo_children():
            widget.destroy()
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=self.font_main, background=self.bg_main, foreground=self.text_primary)
        style.configure('Header.TLabel', font=self.font_header, background=self.bg_main, foreground=self.text_accent)
        style.configure('Info.TLabel', font=self.font_info, background=self.bg_main, foreground=self.text_info)
        style.configure('Legend.TLabel', font=self.font_legend, background=self.bg_main, foreground=self.text_primary)
        style.configure('TFrame', background=self.bg_main)
        style.configure('TLabelFrame', background=self.bg_main, foreground=self.text_primary, borderwidth=0, relief='flat')
        style.configure('TLabelFrame.Label', background=self.bg_main, foreground=self.text_primary, font=self.font_main)
        style.configure('Modern.TButton', font=self.font_header, padding=(20, 10), background=self.bg_secondary, foreground=self.text_primary, borderwidth=0, relief='flat')
        style.map('Modern.TButton', background=[('active', self.text_accent), ('pressed', self.text_accent)])
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 15))
        self.info_label = ttk.Label(top_frame, text="", style='Title.TLabel')
        self.info_label.pack(side=tk.LEFT)
        button_frame = ttk.Frame(top_frame)
        button_frame.pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="◀ Previous", command=self._previous_text, style='Modern.TButton').pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_frame, text="Next ▶", command=self._next_text, style='Modern.TButton').pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_frame, text="🎲 Random", command=self._random_text, style='Modern.TButton').pack(side=tk.LEFT, padx=(0, 8))
        self.mode_toggle = ttk.Button(button_frame, text="🌙" if self.dark_mode else "☀️", command=self._toggle_mode, style='Modern.TButton')
        self.mode_toggle.pack(side=tk.LEFT, padx=(0, 8))
        legend_frame = ttk.LabelFrame(main_frame, text="Categories", padding=15)
        legend_frame.pack(fill=tk.X, pady=(0, 0))
        legend_canvas = tk.Canvas(legend_frame, bg=self.bg_main, highlightthickness=0, bd=0, height=120)
        legend_canvas.pack(fill=tk.X, expand=False)
        legend_inner = ttk.Frame(legend_canvas)
        legend_window = legend_canvas.create_window((0, 0), window=legend_inner, anchor='nw')
        n_cats = len(self.categories)
        n_cols = min(5, max(3, (n_cats + 7) // 8))
        self.legend_labels = {}
        self.category_snippet_positions = {cat: [] for cat in self.categories}
        self.category_snippet_indices = {cat: 0 for cat in self.categories}
        for i, (category, color) in enumerate(self.category_colors.items()):
            row = i // n_cols
            col = i % n_cols
            category_frame = ttk.Frame(legend_inner)
            category_frame.grid(row=row, column=col, padx=18, pady=6, sticky='w')
            color_canvas = tk.Canvas(category_frame, width=38, height=24, bg=self.bg_main, highlightthickness=0, bd=0)
            color_canvas.pack(side=tk.LEFT, padx=(0, 12))
            color_canvas.create_rectangle(4, 4, 34, 20, fill=color, outline=self.legend_border, width=2)
            legend_label = ttk.Label(category_frame, text=f"{category.replace('_', ' ').title()} (0)", style='Legend.TLabel', cursor="hand2")
            legend_label.pack(side=tk.LEFT)
            legend_label.bind('<Button-1>', lambda e, cat=category: self._find_next_snippet(cat))
            legend_label.bind('<Enter>', lambda e, lbl=legend_label: lbl.config(foreground=self.text_accent))
            legend_label.bind('<Leave>', lambda e, lbl=legend_label: lbl.config(foreground=self.text_primary))
            self.legend_labels[category] = legend_label
        legend_inner.update_idletasks()
        legend_canvas.config(scrollregion=legend_canvas.bbox("all"))
        if legend_inner.winfo_reqwidth() > legend_canvas.winfo_width():
            legend_canvas.config(width=legend_inner.winfo_reqwidth())
        # Add separator line
        sep = tk.Frame(main_frame, height=2, bg=self.legend_border)
        sep.pack(fill=tk.X, pady=(0, 0))
        text_frame = ttk.LabelFrame(main_frame, text="Text Content", padding=15)
        text_frame.pack(fill=tk.BOTH, expand=True)
        self.text_widget = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            font=self.font_text,
            bg='#23272a' if self.dark_mode else '#ffffff',
            fg='#f7f7f7' if self.dark_mode else '#000000',
            relief=tk.FLAT,
            borderwidth=2,
            padx=15,
            pady=15,
            selectbackground=self.text_accent,
            selectforeground='#23272a' if self.dark_mode else '#ffffff',
            insertbackground='#f7f7f7' if self.dark_mode else '#000000',
            spacing1=4,
            spacing3=4
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        for category, color in self.category_colors.items():
            self.text_widget.tag_configure(
                category,
                background=color,
                foreground='#23272a' if self.dark_mode else '#000000',
                relief=tk.RAISED,
                borderwidth=1,
                font=self.font_text
            )
            # Emphasis tag for selected snippet
            self.text_widget.tag_configure(f"{category}_emph", background=color, borderwidth=4, relief=tk.SOLID)
        for i, category in enumerate(self.categories):
            self.text_widget.tag_raise(category)
        self.snippet_info = ttk.Label(main_frame, text="", style='Info.TLabel')
        self.snippet_info.pack(pady=(10, 0))
        self.text_widget.bind('<Motion>', self._on_mouse_motion)
        self.text_widget.bind('<Leave>', self._on_mouse_leave)

    def _toggle_mode(self):
        self.dark_mode = not self.dark_mode
        self._apply_theme()
        self._display_current_text()

    def _letters_only(self, text: str) -> str:
        """Keep only lowercase letters a-z, remove everything else."""
        if not text:
            return ""
        return re.sub(r'[^a-z]', '', text.lower())

    def _find_text_position(self, text: str, snippet: str) -> tuple:
        """Robust text position finding using the same logic as codify."""
        clean_snippet = snippet.strip()
        if not clean_snippet:
            return None, None
        
        # Strategy 1: Direct exact match
        start = text.find(clean_snippet)
        if start != -1:
            return start, start + len(clean_snippet)
        
        # Strategy 2: Case-insensitive match  
        start = text.lower().find(clean_snippet.lower())
        if start != -1:
            return start, start + len(clean_snippet)
        
        # Strategy 3: Letters-only matching (most robust)
        text_letters = self._letters_only(text)
        snippet_letters = self._letters_only(clean_snippet)
        
        if snippet_letters and snippet_letters in text_letters:
            # Find approximate position using letters-only
            letters_idx = text_letters.find(snippet_letters)
            ratio = letters_idx / len(text_letters) if text_letters else 0
            approx_start = int(ratio * len(text))
            
            # Search in a window around the approximate position
            window_size = len(clean_snippet) * 3
            search_start = max(0, approx_start - window_size)
            search_end = min(len(text), approx_start + window_size)
            search_text = text[search_start:search_end]
            
            # Try to find exact match in this window
            local_pos = self._find_in_window(search_text, clean_snippet)
            if local_pos is not None:
                return search_start + local_pos[0], search_start + local_pos[1]
        
        # Strategy 4: Try with first/last parts for partial matches
        if len(snippet_letters) >= 20:
            # Try first 20 letters
            first_20 = snippet_letters[:20]
            if first_20 in text_letters:
                letters_idx = text_letters.find(first_20)
                ratio = letters_idx / len(text_letters) if text_letters else 0
                approx_start = int(ratio * len(text))
                
                # Search window
                window_size = len(clean_snippet) * 2
                search_start = max(0, approx_start - window_size//2)
                search_end = min(len(text), approx_start + window_size)
                search_text = text[search_start:search_end]
                
                local_pos = self._find_in_window(search_text, clean_snippet)
                if local_pos is not None:
                    return search_start + local_pos[0], search_start + local_pos[1]
        
        # Strategy 5: Fallback to regex (last resort)
        try:
            pattern = re.escape(clean_snippet[:50])
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.start(), match.end()
        except:
            pass
        
        return None, None

    def _find_in_window(self, window_text: str, target: str) -> tuple:
        """Find target in window using multiple strategies."""
        # Direct match
        idx = window_text.find(target)
        if idx != -1:
            return idx, idx + len(target)
        
        # Case insensitive
        idx = window_text.lower().find(target.lower())
        if idx != -1:
            return idx, idx + len(target)
        
        # Try with some normalization
        normalized_window = re.sub(r'\s+', ' ', window_text.lower())
        normalized_target = re.sub(r'\s+', ' ', target.lower())
        
        idx = normalized_window.find(normalized_target)
        if idx != -1:
            return idx, idx + len(normalized_target)
        
        return None

    def _display_current_text(self):
        if self.current_index >= len(self.df):
            self.current_index = 0
        row = self.df.iloc[self.current_index]
        text = str(row[self.column_name])
        additional_info = ""
        if 'conversation_id' in self.df.columns:
            additional_info = f" | ID: {row['conversation_id']}"
        elif 'id' in self.df.columns:
            additional_info = f" | ID: {row['id']}"
        self.info_label.config(
            text=f"Text {self.current_index + 1} of {len(self.df)}{additional_info}"
        )
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(1.0, text)
        highlights = []
        snippet_count = 0
        self.category_snippet_positions = {cat: [] for cat in self.categories}
        if self.dynamic_mode:
            coded_passages = row['coded_passages'] if 'coded_passages' in row else {}
            for category in self.categories:
                if category in coded_passages and coded_passages[category]:
                    snippets = coded_passages[category]
                    if isinstance(snippets, list):
                        for snippet in snippets:
                            if snippet and isinstance(snippet, str):
                                start_pos, end_pos = self._find_text_position(text, snippet)
                                if start_pos is not None:
                                    highlights.append({
                                        'start': start_pos,
                                        'end': end_pos,
                                        'category': category,
                                        'snippet': snippet
                                    })
                                    self.category_snippet_positions[category].append((start_pos, end_pos))
                                    snippet_count += 1
        else:
            for category in self.categories:
                if category in row and row[category]:
                    snippets = row[category]
                    if isinstance(snippets, list):
                        for snippet in snippets:
                            if snippet and isinstance(snippet, str):
                                start_pos, end_pos = self._find_text_position(text, snippet)
                                if start_pos is not None:
                                    highlights.append({
                                        'start': start_pos,
                                        'end': end_pos,
                                        'category': category,
                                        'snippet': snippet
                                    })
                                    self.category_snippet_positions[category].append((start_pos, end_pos))
                                    snippet_count += 1
        highlights.sort(key=lambda x: x['start'])
        self.text_widget.tag_remove('highlight', '1.0', tk.END)
        for tag in self.text_widget.tag_names():
            if tag.startswith('hover_') or tag.endswith('_emph'):
                self.text_widget.tag_delete(tag)
        for highlight in highlights:
            start_idx = f"1.0+{highlight['start']}c"
            end_idx = f"1.0+{highlight['end']}c"
            tag_name = f"{highlight['category']}_{highlight['start']}_{highlight['end']}"
            self.text_widget.tag_add(tag_name, start_idx, end_idx)
            color = self.category_colors[highlight['category']]
            self.text_widget.tag_configure(
                tag_name,
                background=color,
                foreground='#23272a' if self.dark_mode else '#000000',
                relief=tk.RAISED,
                borderwidth=1,
                font=self.font_text
            )
        self.text_widget.config(state=tk.DISABLED)
        self.current_highlights = highlights
        self._update_legend_counts()
        self._update_snippet_info()
        self.selected_snippet_tag = None

    def _update_legend_counts(self):
        for cat, label in self.legend_labels.items():
            count = len(self.category_snippet_positions.get(cat, []))
            label.config(text=f"{cat.replace('_', ' ').title()} ({count})")

    def _find_next_snippet(self, category):
        positions = self.category_snippet_positions.get(category, [])
        if not positions:
            return  # No snippets for this category
        idx = self.category_snippet_indices.get(category, 0)
        start, end = positions[idx]
        start_idx = f"1.0+{start}c"
        end_idx = f"1.0+{end}c"
        self.text_widget.tag_remove('sel', '1.0', tk.END)
        # Remove previous emphasis
        if self.selected_snippet_tag:
            self.text_widget.tag_remove(self.selected_snippet_tag, '1.0', tk.END)
        # Add emphasis to the selected snippet
        emph_tag = f"{category}_emph"
        self.text_widget.tag_add(emph_tag, start_idx, end_idx)
        self.text_widget.tag_raise(emph_tag)
        self.selected_snippet_tag = emph_tag
        self.text_widget.see(start_idx)
        idx = (idx + 1) % len(positions)
        self.category_snippet_indices[category] = idx

    def _update_snippet_info(self):
        if not hasattr(self, 'current_highlights'):
            self.snippet_info.config(text="")
            return
        category_counts = {}
        for highlight in self.current_highlights:
            cat = highlight['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        if category_counts:
            count_text = ", ".join([f"{cat.replace('_', ' ').title()}: {count}" for cat, count in category_counts.items()])
            self.snippet_info.config(text=f"Highlighted snippets - {count_text}")
        else:
            self.snippet_info.config(text="No snippets found for this text")

    def _next_text(self):
        self.current_index = (self.current_index + 1) % len(self.df)
        self._display_current_text()
    
    def _previous_text(self):
        self.current_index = (self.current_index - 1) % len(self.df)
        self._display_current_text()
    
    def _random_text(self):
        self.current_index = random.randint(0, len(self.df) - 1)
        self._display_current_text()
    
    def show(self):
        self.root.mainloop()
    
    def destroy(self):
        self.root.destroy()

    def _on_mouse_motion(self, event):
        index = self.text_widget.index(f"@{event.x},{event.y}")
        pos = self.text_widget.count('1.0', index, 'chars')[0]
        hovered = []
        for highlight in self.current_highlights:
            if highlight['start'] <= pos < highlight['end']:
                hovered.append(highlight)
        if hovered:
            cats = sorted(set(h['category'] for h in hovered))
            if cats != self.last_tooltip_cats:
                # Show each category in its highlight color
                label_text = " | ".join([
                    f"\u25A0 {cat.replace('_', ' ').title()}" for cat in cats
                ])
                self._show_tooltip(event.x_root, event.y_root, cats)
                self.last_tooltip_cats = cats
        else:
            self._hide_tooltip()
            self.last_tooltip_cats = None

    def _on_mouse_leave(self, event):
        self._hide_tooltip()
        self.last_tooltip_cats = None

    def _show_tooltip(self, x, y, cats):
        if self.tooltip:
            self.tooltip.destroy()
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x+20}+{y+20}")
        frame = tk.Frame(self.tooltip, bg='#23272a' if self.dark_mode else '#f7f7f7', bd=0, highlightthickness=0)
        frame.pack()
        for cat in cats:
            color = self.category_colors.get(cat, '#00bcd4')
            label = tk.Label(frame, text=cat.replace('_', ' ').title(), font=self.font_popup,
                             background='#23272a' if self.dark_mode else '#f7f7f7',
                             foreground=color, padx=18, pady=8, borderwidth=0)
            label.pack(anchor='w')
        # Drop shadow effect
        self.tooltip.lift()
        self.tooltip.attributes('-topmost', True)
        try:
            self.tooltip.attributes('-alpha', 0.98)
        except Exception:
            pass

    def _hide_tooltip(self):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

_COLAB_STYLE = """
<style>
.gabriel-codify-viewer {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #f5f7fa;
    background: transparent;
}
.gabriel-codify-viewer .gabriel-status {
    font-size: 14px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.7);
    margin-bottom: 8px;
}
.gabriel-codify-viewer .gabriel-controls {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 12px;
}
.gabriel-codify-viewer .gabriel-passage-panel {
    background: #13161a;
    border: 1px solid #2b323c;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 16px 40px rgba(9, 11, 16, 0.45);
}
.gabriel-codify-viewer .gabriel-passage-scroll {
    max-height: 560px;
    overflow-y: auto;
    padding-right: 12px;
}
.gabriel-codify-viewer .gabriel-legend {
    position: sticky;
    top: 0;
    z-index: 2;
    background: #13161a;
    padding-bottom: 12px;
    margin-bottom: 16px;
    border-bottom: 1px solid #2b323c;
}
.gabriel-codify-viewer .gabriel-legend-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}
.gabriel-codify-viewer .gabriel-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    font-size: 13px;
    color: rgba(255, 255, 255, 0.88);
    cursor: pointer;
    transition: background 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
    text-decoration: none;
    font: inherit;
    line-height: 1.2;
}
.gabriel-codify-viewer .gabriel-legend-item:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(255, 255, 255, 0.18);
    transform: translateY(-1px);
}
.gabriel-codify-viewer .gabriel-legend-item:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.65);
}
.gabriel-codify-viewer .gabriel-legend-item span {
    pointer-events: none;
}
.gabriel-codify-viewer .gabriel-legend-color {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid rgba(0, 0, 0, 0.18);
}
.gabriel-codify-viewer .gabriel-legend-label {
    font-weight: 600;
    color: inherit;
}
.gabriel-codify-viewer .gabriel-legend-count {
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    color: rgba(255, 255, 255, 0.78);
}
.gabriel-codify-viewer .gabriel-header {
    margin-bottom: 14px;
    padding: 14px 16px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.gabriel-codify-viewer .gabriel-header-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 6px;
}
.gabriel-codify-viewer .gabriel-header-label {
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.05em;
    color: rgba(255, 255, 255, 0.7);
}
.gabriel-codify-viewer .gabriel-header-value {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.94);
}
.gabriel-codify-viewer .gabriel-active-cats {
    margin-top: 4px;
    font-size: 13px;
    color: rgba(255, 255, 255, 0.85);
}
.gabriel-codify-viewer .gabriel-text {
    font-size: 15px;
    line-height: 1.7;
    color: rgba(245, 247, 250, 0.96);
}
.gabriel-codify-viewer .gabriel-text p {
    margin: 0 0 1em 0;
}
.gabriel-codify-viewer .gabriel-snippet {
    position: relative;
    border-radius: 6px;
    padding: 1px 5px;
    font-weight: 600;
    color: #0d1014;
    cursor: pointer;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.gabriel-codify-viewer .gabriel-snippet::after {
    content: attr(data-label);
    position: absolute;
    left: 0;
    bottom: 100%;
    transform: translateY(-6px);
    background: rgba(8, 11, 17, 0.92);
    color: #f8fafc;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease, transform 0.2s ease;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
    z-index: 5;
}
.gabriel-codify-viewer .gabriel-snippet:hover::after {
    opacity: 1;
    transform: translateY(-10px);
}
.gabriel-codify-viewer .gabriel-snippet-active {
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.8), 0 0 18px rgba(255, 255, 255, 0.35);
}
.gabriel-codify-viewer .gabriel-empty {
    font-style: italic;
    color: rgba(255, 255, 255, 0.65);
}
@media (prefers-color-scheme: light) {
    .gabriel-codify-viewer {
        color: #1f2933;
    }
    .gabriel-codify-viewer .gabriel-passage-panel {
        background: #f7f9fb;
        border-color: #d0d7e2;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12);
    }
    .gabriel-codify-viewer .gabriel-legend {
        background: #f7f9fb;
        border-color: #d0d7e2;
    }
    .gabriel-codify-viewer .gabriel-header {
        background: rgba(15, 23, 42, 0.06);
        border-color: rgba(15, 23, 42, 0.12);
    }
    .gabriel-codify-viewer .gabriel-header-label {
        color: rgba(15, 23, 42, 0.65);
    }
    .gabriel-codify-viewer .gabriel-header-value {
        color: rgba(15, 23, 42, 0.92);
    }
    .gabriel-codify-viewer .gabriel-text {
        color: #1f2933;
    }
    .gabriel-codify-viewer .gabriel-legend-item {
        background: rgba(15, 23, 42, 0.06);
        border-color: rgba(15, 23, 42, 0.12);
        color: rgba(15, 23, 42, 0.82);
    }
    .gabriel-codify-viewer .gabriel-legend-item:hover {
        background: rgba(15, 23, 42, 0.1);
        border-color: rgba(15, 23, 42, 0.18);
    }
    .gabriel-codify-viewer .gabriel-legend-item:focus-visible {
        box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.4);
    }
    .gabriel-codify-viewer .gabriel-legend-count {
        background: rgba(15, 23, 42, 0.1);
        color: rgba(15, 23, 42, 0.75);
    }
    .gabriel-codify-viewer .gabriel-empty {
        color: rgba(15, 23, 42, 0.55);
    }
    .gabriel-codify-viewer .gabriel-snippet::after {
        background: rgba(15, 23, 42, 0.92);
        color: #f8fafc;
    }
    .gabriel-codify-viewer .gabriel-snippet-active {
        box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.28), 0 0 18px rgba(15, 23, 42, 0.3);
    }
}
</style>
<script>
(function () {
    if (window.__gabrielPassageViewerEnhancer) {
        return;
    }
    window.__gabrielPassageViewerEnhancer = true;

    const stateMap = new WeakMap();

    function ensureState(container, token) {
        let record = stateMap.get(container);
        if (!record || record.token !== token) {
            record = { token: token, indices: {} };
            stateMap.set(container, record);
        }
        return record;
    }

    function escapeSelector(value) {
        if (window.CSS && typeof window.CSS.escape === 'function') {
            return window.CSS.escape(value);
        }
        return String(value).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
    }

    function bindLegendItem(item) {
        if (!(item instanceof Element) || item.dataset.gabrielBound === '1') {
            return;
        }
        const legend = item.closest('.gabriel-legend');
        const container = item.closest('.gabriel-codify-viewer');
        if (!legend || !container) {
            return;
        }
        const category = item.getAttribute('data-category');
        if (!category) {
            return;
        }
        const token = legend.getAttribute('data-legend-token') || '';
        const state = ensureState(container, token);
        item.dataset.gabrielBound = '1';
        item.addEventListener('click', function (event) {
            event.preventDefault();
            const selector = '.gabriel-snippet[data-category="' + escapeSelector(category) + '"]';
            const snippets = container.querySelectorAll(selector);
            if (!snippets.length) {
                return;
            }
            const nextIndex = state.indices[category] || 0;
            const target = snippets[nextIndex % snippets.length];
            state.indices[category] = (nextIndex + 1) % snippets.length;
            container.querySelectorAll('.gabriel-snippet.gabriel-snippet-active').forEach(function (el) {
                if (el !== target) {
                    el.classList.remove('gabriel-snippet-active');
                }
            });
            target.classList.add('gabriel-snippet-active');
            if (typeof target.scrollIntoView === 'function') {
                target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            window.setTimeout(function () {
                target.classList.remove('gabriel-snippet-active');
            }, 1600);
        });
    }

    function scan(root) {
        if (!(root instanceof Element)) {
            return;
        }
        root.querySelectorAll('.gabriel-legend-item').forEach(bindLegendItem);
    }

    const observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            mutation.addedNodes.forEach(function (node) {
                if (!(node instanceof Element)) {
                    return;
                }
                if (node.classList.contains('gabriel-legend-item')) {
                    bindLegendItem(node);
                } else {
                    scan(node);
                }
            });
        });
    });

    if (document && document.body) {
        observer.observe(document.body, { childList: true, subtree: true });
        scan(document.body);
    }
})();
</script>
"""


def _normalize_header_columns(
    header_columns: Optional[Union[Sequence[Any], Any]]
) -> List[Tuple[str, str]]:
    if header_columns is None:
        return []

    if isinstance(header_columns, (str, bytes)):
        header_sequence: Iterable[Any] = [header_columns]
    elif isinstance(header_columns, Iterable):
        header_sequence = header_columns
    else:
        header_sequence = [header_columns]

    normalized: List[Tuple[str, str]] = []
    for entry in header_sequence:
        if isinstance(entry, (list, tuple)) and entry:
            column = str(entry[0])
            label = str(entry[1]) if len(entry) > 1 else column
        else:
            column = str(entry)
            label = column
        pretty_label = label.replace("_", " ").title()
        normalized.append((column, pretty_label))
    return normalized


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    return False


def _format_header_value(value: Any) -> str:
    if _is_na(value):
        return ""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts)
    return str(value)


def _build_highlighted_text(
    text: str,
    snippet_map: Dict[str, List[str]],
    category_colors: Dict[str, str],
) -> str:
    if not text:
        return "<div class='gabriel-empty'>No text available.</div>"

    spans: List[Tuple[int, int, str]] = []
    for category, snippets in snippet_map.items():
        if not snippets or category not in category_colors:
            continue
        for snippet in snippets:
            if not snippet:
                continue
            start = 0
            while True:
                index = text.find(snippet, start)
                if index == -1:
                    break
                spans.append((index, index + len(snippet), category))
                start = index + len(snippet)

    if not spans:
        return html.escape(text).replace("\n", "<br/>")

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    merged: List[Tuple[int, int, str]] = []
    current_end = -1
    for start, end, category in spans:
        if start < current_end:
            continue
        merged.append((start, end, category))
        current_end = end

    pieces: List[str] = []
    cursor = 0
    snippet_indices: Dict[str, int] = {}
    for start, end, category in merged:
        pieces.append(html.escape(text[cursor:start]).replace("\n", "<br/>"))
        snippet_html = html.escape(text[start:end]).replace("\n", "<br/>")
        category_key = str(category)
        label = html.escape(category_key.replace("_", " ").title())
        color = category_colors.get(category_key, "#ffd54f")
        safe_color = html.escape(color, quote=True)
        safe_category = html.escape(category_key, quote=True)
        index = snippet_indices.get(category_key, 0)
        snippet_indices[category_key] = index + 1
        slug = re.sub(r"[^0-9a-zA-Z_-]+", "-", category_key).strip("-")
        if not slug:
            slug = "category"
        element_id = f"gabriel-snippet-{slug}-{index}"
        pieces.append(
            "<span class='gabriel-snippet' "
            f"data-category='{safe_category}' data-index='{index}' "
            f"data-label='{label}' id='{element_id}' "
            f"style='background-color:{safe_color}' title='{label}'>"
            f"{snippet_html}</span>"
        )
        cursor = end
    pieces.append(html.escape(text[cursor:]).replace("\n", "<br/>"))
    return "".join(pieces)


def _build_header_html(
    header_rows: List[Tuple[str, str]],
    active_categories: List[str],
) -> str:
    if not header_rows and not active_categories:
        return ""

    parts: List[str] = []
    for label, value in header_rows:
        safe_label = html.escape(label)
        safe_value = html.escape(value).replace("\n", "<br/>")
        parts.append(
            f"<div class='gabriel-header-row'>"
            f"<span class='gabriel-header-label'>{safe_label}:</span>"
            f"<span class='gabriel-header-value'>{safe_value}</span>"
            f"</div>"
        )

    if active_categories:
        active = ", ".join(
            html.escape(cat.replace("_", " ").title()) for cat in active_categories
        )
        parts.append(
            "<div class='gabriel-active-cats'><strong>Categories:</strong> "
            f"{active}</div>"
        )

    return "<div class='gabriel-header'>" + "".join(parts) + "</div>"


def _build_legend_html(
    category_colors: Dict[str, str],
    category_counts: Dict[str, int],
    legend_token: Optional[str] = None,
) -> str:
    if not category_colors:
        return (
            "<div class='gabriel-legend gabriel-empty'>No categories to display.</div>"
        )

    items = []
    for category, color in category_colors.items():
        pretty = category.replace("_", " ").title()
        label = html.escape(pretty)
        raw_count = category_counts.get(category, 0)
        try:
            count_value = int(raw_count)
        except (TypeError, ValueError):
            count_value = 0
        aria_label = html.escape(f"{pretty} ({count_value})", quote=True)
        count = html.escape(str(count_value))
        safe_color = html.escape(color, quote=True)
        safe_category = html.escape(category, quote=True)
        items.append(
            "<button type='button' class='gabriel-legend-item' "
            f"data-category='{safe_category}' data-count='{count}' aria-label='{aria_label}'>"
            f"<span class='gabriel-legend-color' style='background:{safe_color}'></span>"
            f"<span class='gabriel-legend-label'>{label}</span>"
            f"<span class='gabriel-legend-count'>{count}</span>"
            "</button>"
        )

    token_attr = (
        f" data-legend-token='{html.escape(legend_token, quote=True)}'"
        if legend_token
        else ""
    )
    return (
        f"<div class='gabriel-legend'{token_attr}><div class='gabriel-legend-grid'>"
        + "".join(items)
        + "</div></div>"
    )


def _view_coded_passages_colab(
    df: pd.DataFrame,
    column_name: str,
    categories: Optional[Union[List[str], str]] = None,
    header_columns: Optional[Union[Sequence[Any], Any]] = None,
) -> None:
    """Display passages inside a Jupyter notebook.

    This simplified viewer avoids any desktop GUI requirements, making it
    suitable for headless environments such as Google Colab. Passages are
    rendered with HTML highlighting directly in the notebook output.
    """

    from IPython.display import HTML, display  # pragma: no cover - optional

    df = df.copy()

    # Detect categories in the same way as :class:`PassageViewer`.
    if categories is None and "coded_passages" in df.columns:
        dynamic_mode = True
        all_categories = set()
        for coded_passages in df["coded_passages"]:
            if coded_passages and isinstance(coded_passages, dict):
                all_categories.update(coded_passages.keys())
        categories = sorted(list(all_categories))
    elif isinstance(categories, str) and categories == "coded_passages":
        dynamic_mode = True
        all_categories = set()
        for coded_passages in df["coded_passages"]:
            if coded_passages and isinstance(coded_passages, dict):
                all_categories.update(coded_passages.keys())
        categories = sorted(list(all_categories))
    else:
        dynamic_mode = False
        categories = categories if categories else []

    normalized_headers = _normalize_header_columns(header_columns)
    colors = _generate_distinct_colors(len(categories))
    category_colors = dict(zip(categories, colors))

    passages: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        raw_text = row.get(column_name)
        text = "" if _is_na(raw_text) else str(raw_text)

        if dynamic_mode:
            raw_map = row.get("coded_passages")
            snippet_source = raw_map if isinstance(raw_map, dict) else {}
        else:
            snippet_source = {cat: row.get(cat, []) for cat in categories}

        snippet_map: Dict[str, List[str]] = {cat: [] for cat in categories}
        for cat, snippets in snippet_source.items():
            if cat not in category_colors:
                continue
            cleaned: List[str]
            if isinstance(snippets, str):
                cleaned = [snippets]
            elif isinstance(snippets, (list, tuple, set)):
                cleaned = [
                    str(item)
                    for item in snippets
                    if item is not None and not _is_na(item) and str(item)
                ]
            elif _is_na(snippets) or snippets is None:
                cleaned = []
            elif snippets:
                cleaned = [str(snippets)]
            else:
                cleaned = []
            snippet_map[cat] = cleaned

        header_rows: List[Tuple[str, str]] = []
        for column, label in normalized_headers:
            value = row.get(column)
            formatted = _format_header_value(value)
            if formatted:
                header_rows.append((label, formatted))

        active_categories = [cat for cat, snippets in snippet_map.items() if snippets]
        passage_counts = {
            cat: len(snippet_map.get(cat, []))
            for cat in categories
        }
        passages.append(
            {
                "text": text,
                "snippets": snippet_map,
                "header": header_rows,
                "active": active_categories,
                "counts": passage_counts,
            }
        )
    total = len(passages)

    try:  # pragma: no cover - optional dependency
        import ipywidgets as widgets  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        widgets = None  # type: ignore

    if widgets is not None:
        display(HTML(_COLAB_STYLE))

        if total == 0:
            display(
                widgets.HTML(
                    "<div class='gabriel-codify-viewer gabriel-empty'>No passages to display.</div>"
                )
            )
            return

        status = widgets.HTML()
        slider = widgets.IntSlider(
            min=1,
            max=total,
            value=1,
            description="Passage",
            continuous_update=False,
        )
        slider.layout = widgets.Layout(width="100%")

        prev_button = widgets.Button(description="◀ Previous")
        next_button = widgets.Button(description="Next ▶")
        random_button = widgets.Button(description="🎲 Random")

        controls_box = widgets.HBox([prev_button, next_button, random_button, slider])
        try:
            controls_box.add_class("gabriel-controls")
        except Exception:  # pragma: no cover - best effort styling
            pass

        passage_display = widgets.HTML()
        passage_display.layout = widgets.Layout(width="100%")

        current = {"idx": 0}

        def _render(index: int) -> None:
            if total == 0:
                return
            index = max(0, min(total - 1, index))
            current["idx"] = index
            payload = passages[index]
            body_html = _build_highlighted_text(
                payload["text"], payload["snippets"], category_colors
            )
            header_html = _build_header_html(payload["header"], payload["active"])
            legend_token = f"interactive-{index}-{random.random()}"
            legend_html = _build_legend_html(
                category_colors, payload["counts"], legend_token
            )
            passage_html = (
                "<div class='gabriel-codify-viewer'><div class='gabriel-passage-panel'>"
                "<div class='gabriel-passage-scroll'>"
                f"{legend_html}{header_html}<div class='gabriel-text'>{body_html}</div>"
                "</div></div></div>"
            )
            passage_display.value = passage_html
            status.value = (
                "<div class='gabriel-codify-viewer'><div class='gabriel-status'>Passage "
                f"<strong>{index + 1}</strong> of {total}</div></div>"
            )
            if slider.value != index + 1:
                slider.value = index + 1

        def _prev(_event: Any) -> None:
            new_index = (current["idx"] - 1) % total
            _render(new_index)

        def _next(_event: Any) -> None:
            new_index = (current["idx"] + 1) % total
            _render(new_index)

        def _random(_event: Any) -> None:
            new_index = random.randrange(total)
            _render(new_index)

        def _slider_change(change: Dict[str, Any]) -> None:
            if change.get("name") == "value" and isinstance(change.get("new"), int):
                _render(change["new"] - 1)

        prev_button.on_click(_prev)
        next_button.on_click(_next)
        random_button.on_click(_random)
        slider.observe(_slider_change, names="value")

        ui = widgets.VBox([status, controls_box, passage_display])
        display(ui)
        _render(0)
        return

    html_parts: List[str] = [_COLAB_STYLE, "<div class='gabriel-codify-viewer'>"]
    if total == 0:
        html_parts.append("<div class='gabriel-empty'>No passages to display.</div>")
    else:
        for idx, payload in enumerate(passages):
            legend_token = f"static-{idx}-{random.random()}"
            legend_html = _build_legend_html(
                category_colors, payload["counts"], legend_token
            )
            body_html = _build_highlighted_text(
                payload["text"], payload["snippets"], category_colors
            )
            header_html = _build_header_html(payload["header"], payload["active"])
            html_parts.append("<div class='gabriel-passage-panel' style='margin-bottom:18px'>")
            html_parts.append(
                f"<div class='gabriel-status'>Passage <strong>{idx + 1}</strong> of {total}</div>"
            )
            html_parts.append(
                "<div class='gabriel-passage-scroll'>"
                f"{legend_html}{header_html}<div class='gabriel-text'>{body_html}</div>"
                "</div>"
            )
            html_parts.append("</div>")
    html_parts.append("</div>")
    display(HTML("".join(html_parts)))


def view_coded_passages(
    df: pd.DataFrame,
    column_name: str,
    categories: Optional[Union[List[str], str]] = None,
    colab: bool = True,
    header_columns: Optional[Union[Sequence[Any], Any]] = None,
):
    """View coded passages.

    Parameters
    ----------
    df:
        DataFrame containing the passages.
    column_name:
        Column name in ``df`` holding the raw text.
    categories:
        Either a list of category column names or ``"coded_passages"`` for
        dynamic dictionaries.
    colab:
        When ``True`` (the default), use the lightweight HTML viewer that
        works in Google Colab or other headless notebook environments.
        Passing ``False`` launches the full ``tkinter`` GUI.
    header_columns:
        Optional sequence of column names (or ``(column, label)`` tuples)
        displayed above each passage when using the viewer. The values are
        rendered in the order provided, allowing quick inspection of metadata
        such as speaker names or timestamps.
    """

    if colab:
        _view_coded_passages_colab(
            df,
            column_name,
            categories,
            header_columns=header_columns,
        )
        return None

    viewer = PassageViewer(df, column_name, categories)
    viewer.show()
    return viewer


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Sample data
    sample_data = {
        'id': [1, 2, 3],
        'text': [
            "This is a great example of positive text. I really appreciate your help with this matter.",
            "I can't believe how terrible this service is. This is absolutely unacceptable behavior.",
            "Could you please explain how this works? I'm genuinely curious about the process."
        ],
        'positive_sentiment': [
            ["This is a great example of positive text", "I really appreciate your help"],
            [],
            ["I'm genuinely curious about the process"]
        ],
        'negative_sentiment': [
            [],
            ["I can't believe how terrible this service is", "This is absolutely unacceptable behavior"],
            []
        ],
        'questions': [
            [],
            [],
            ["Could you please explain how this works?"]
        ]
    }
    
    df = pd.DataFrame(sample_data)
    categories = ['positive_sentiment', 'negative_sentiment', 'questions']
    
    view_coded_passages(df, 'text', categories) 