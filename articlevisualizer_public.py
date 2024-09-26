import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import os
from openai import OpenAI
import openai
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Wedge
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
import textwrap
import seaborn as sns  # Import Seaborn

# Set Seaborn style
sns.set_style("darkgrid")  # Options: "darkgrid", "whitegrid", "dark", "white", "ticks"
sns.set_context("talk")      # Options: "paper", "notebook", "talk", "poster"

# Define a custom color palette
COLOR_PALETTE = {
    'background': '#f9f9f9',
    'title': '#333333',
    'text': '#555555',
    'funnel': ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336'],
    'pros': '#4CAF50',
    'cons': '#F44336',
    'circular': '#2196F3',
    'decision_tree': '#FF9800'
}

# Update rcParams for global settings (optional, since Seaborn handles much of this)
plt.rcParams.update({
    'figure.facecolor': COLOR_PALETTE['background'],
    'axes.facecolor': COLOR_PALETTE['background'],
    'axes.edgecolor': COLOR_PALETTE['background'],
    'axes.labelcolor': COLOR_PALETTE['text'],
    'xtick.color': COLOR_PALETTE['text'],
    'ytick.color': COLOR_PALETTE['text'],
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 12,
    'figure.figsize': (10, 6)
})

# Replace this with your OpenAI API key
client = OpenAI(
    api_key="LMSTUDIO",
    base_url="http://localhost:1234/v1",  # Your local server URL
)


def openai_chat_completion(client, model, messages, max_tokens):
    """Makes an OpenAI chat completion API call and logs the request."""
    print("Sending request to OpenAI API...")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

# Function to get data from OpenAI's LLM using ChatCompletion API
def ask_llm(article_text):
    llm_prompt = f"""
    Analyze the following article and provide structured data for various visualizations:
    
    1. Funnel Diagram: Provide the stages of the funnel process as a comma-separated list in the format with a minimum of 5 items:
       Funnel: stage1, stage2, stage3, stage4, stage5
    
    2. Timeline: Extract key milestones and dates in the format and have at least 4:
       Timeline:
       - year: event1 description
       - year: event2 description
    
    3. Pros and Cons Table: Return the pros and cons in the format and have at least two pros and two cons:
       Pros:
       - pro1
       - pro2
       Cons:
       - con1
       - con2
    
    4. Circular Diagram: Provide the core theme and sub-elements in the format with at least 4 elements:
       Circular Diagram:
       Theme: core theme
       Sub-elements: sub-element1, sub-element2, sub-element3, sub-element4
    
    5. Decision Tree: Provide decisions and outcomes as a list in the format and have at least 4:
       Decision Tree:
       - parent1 -> child1
       - parent2 -> child2
    
    Article:
    {article_text}
    """
    
    response = openai_chat_completion(
        client,
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": llm_prompt}],
        max_tokens=2000
    )

    # Save the response to a text file
    with open('openai_response.txt', 'w') as f:
        f.write(response)

    return response

# Function to parse the structured LLM response
def parse_llm_response(response_text):
    data = {}

    # Parse funnel data
    funnel_lines = [line for line in response_text.split("\n") if line.startswith("Funnel:")]
    if funnel_lines:
        data["funnel"] = funnel_lines[0].replace("Funnel:", "").strip().split(", ")
    else:
        data["funnel"] = []  # Handle missing funnel case appropriately

    # Parse timeline data
    try:
        timeline_start = response_text.index("Timeline:")
        pros_start = response_text.index("Pros:")
        timeline_section = response_text[timeline_start:pros_start].strip().split("\n")[1:]
        data["timeline"] = [tuple(item.split(": ", 1)) for item in timeline_section if ": " in item]
    except ValueError:
        data["timeline"] = []  # Handle missing timeline case

    # Parse pros and cons data
    try:
        pros_start = response_text.index("Pros:")
        cons_start = response_text.index("Cons:")
        pros_section = response_text[pros_start:cons_start].strip().split("\n")[1:]
        cons_section = response_text[cons_start:].strip().split("\n")[1:]
        data["pros"] = [pro.strip().strip('- ') for pro in pros_section if pro.strip()]
        data["cons"] = [con.strip().strip('- ') for con in cons_section if con.strip()]
    except ValueError:
        data["pros"], data["cons"] = [], []  # Handle missing pros/cons

    # Parse circular diagram data
    try:
        circular_start = response_text.index("Circular Diagram:")
        decision_tree_start = response_text.index("Decision Tree:")
        circular_section = response_text[circular_start:decision_tree_start].strip().split("\n")
        theme_line = next(line for line in circular_section if "Theme:" in line)
        sub_elements_line = next(line for line in circular_section if "Sub-elements:" in line)
        data["circular_theme"] = theme_line.replace("Theme:", "").strip()
        data["circular_sub_elements"] = [elem.strip() for elem in sub_elements_line.replace("Sub-elements:", "").split(",")]
    except (ValueError, StopIteration):
        data["circular_theme"], data["circular_sub_elements"] = "", []  # Handle missing circular diagram

    # Parse decision tree data
    try:
        decision_tree_start = response_text.index("Decision Tree:")
        decision_tree_section = response_text[decision_tree_start:].strip().split("\n")[1:]
        data["decision_tree"] = [tuple(item.replace('- ', '').split(" -> ")) for item in decision_tree_section if " -> " in item]
    except ValueError:
        data["decision_tree"] = []  # Handle missing decision tree

    return data

# Visualization for Funnel Diagram

def plot_funnel(funnel_data):
    if not funnel_data:
        print("No funnel data available.")
        return

    stages = [stage.strip() for stage in funnel_data]
    num_stages = len(stages)
    print("Funnel Stages:", stages)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Load a playful font (ensure the font file is accessible)
    label_font = FontProperties(fname='PatrickHand-Regular.ttf', size=12)

    # Define the funnel widths at each stage (from wide to narrow)
    top_width = 8  # Width of the top stage
    bottom_width = 2  # Width of the bottom stage
    heights = 1  # Height of each stage

    # Calculate the widths of each stage to create a funnel shape
    widths = np.linspace(top_width, bottom_width, num_stages + 1)

    # Define colors for the stages (using a gradient)
    colors = plt.cm.Blues(np.linspace(0.3, 0.7, num_stages))

    # Loop through each stage to draw the funnel layers
    for i in range(num_stages):
        # Coordinates for the trapezoid (stage)
        left = (widths[i] - widths[i + 1]) / 2
        right = left + widths[i + 1]
        top = i * heights
        bottom = (i + 1) * heights

        # Define the vertices of the trapezoid
        vertices = [
            [left, top],
            [left + widths[i], top],
            [right, bottom],
            [left, bottom]
        ]

        # Create the polygon (trapezoid)
        polygon = Polygon(vertices, closed=True, facecolor=colors[i], edgecolor='white')
        ax.add_patch(polygon)

        # Add text labels along the flow of the funnel
        text_x = (left + right) / 2
        text_y = top + heights / 2

        ax.text(text_x, text_y, stages[i], ha='center', va='center', fontsize=12,
                color='white', fontproperties=label_font)

        # Add icons to each stage
        icon_path = f'funnel_icon_{i}.png'  # Ensure you have icons named accordingly
        if os.path.exists(icon_path):
            img = mpimg.imread(icon_path)
            imagebox = OffsetImage(img, zoom=0.1)
            ab = AnnotationBbox(imagebox, (text_x - widths[i + 1] / 2 + 0.5, text_y), frameon=False)
            ax.add_artist(ab)

    # Remove axes and set limits
    ax.axis('off')
    ax.set_xlim(0, top_width)
    ax.set_ylim(0, num_stages * heights)

    # Add a title
    plt.title('Funnel Diagram', fontsize=16, fontproperties=label_font)

    plt.tight_layout()
    plt.savefig('funnel_diagram.jpg', dpi=300, bbox_inches='tight')
    


# Visualization for Pros and Cons
def plot_pros_cons(pros, cons):
    if not pros and not cons:
        print("No pros and cons data available.")
        return

    # Determine the maximum number of items to set the figure height dynamically
    max_items = max(len(pros), len(cons))
    fig_height = max(6, max_items * 1)  # Adjust the multiplier as needed for spacing

    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Set font properties
    try:
        title_font = FontProperties(fname='PatrickHand-Regular.ttf', size=20)
        header_font = FontProperties(fname='PatrickHand-Regular.ttf', size=16, weight='bold')
        text_font = FontProperties(fname='PatrickHand-Regular.ttf', size=12)
    except:
        # Fallback to default font if custom font is not found
        title_font = {'size': 20, 'weight': 'bold'}
        header_font = {'size': 16, 'weight': 'bold'}
        text_font = {'size': 12}

    # Simplify color scheme
    bg_color = 'white'
    pros_color = '#4CAF50'  # Green for pros
    cons_color = '#F44336'  # Red for cons
    divider_color = 'gray'

    # Clear axes and set background
    ax.axis('off')
    fig.patch.set_facecolor(bg_color)

    # Add central divider with "VS"
    divider_x = 0.5
    ax.axvline(divider_x, color=divider_color, linewidth=2, ymin=0.05, ymax=0.95)

    # Add "VS" label
    ax.text(divider_x, 0.5, 'VS', ha='center', va='center', fontsize=24, fontproperties=title_font, color=divider_color)

    # Add Pros header
    ax.text(0.25, 1, 'Pros', ha='center', va='center', fontsize=16,
            fontproperties=header_font, color=pros_color, transform=ax.transAxes)

    # Add Cons header
    ax.text(0.75, 1, 'Cons', ha='center', va='center', fontsize=16,
            fontproperties=header_font, color=cons_color, transform=ax.transAxes)

    # Calculate vertical positions based on the number of items
    # Leave some padding at the top and bottom
    padding = 0.1
    available_height = 1 - 2 * padding
    if max_items > 1:
        step = available_height / (max_items - 1)
    else:
        step = 0

    # Function to add items (pros or cons)
    def add_items(items, side):
        for i, item in enumerate(items):
            # Calculate y position
            y = 1 - padding - i * step

            # Set x position based on side
            if side == 'pros':
                x_icon = 0.25 - 0.1
                x_text = 0.2 
                color = pros_color
                icon_prefix = 'pro_icon_'
            else:
                x_icon = 0.75 + 0.1
                x_text = 0.8
                color = cons_color
                icon_prefix = 'con_icon_'

            # Add icon
            icon_path = f'{icon_prefix}{i}.png'  # Ensure you have icons named accordingly
            if os.path.exists(icon_path):
                try:
                    img = mpimg.imread(icon_path)
                    imagebox = OffsetImage(img, zoom=0.05)
                    ab = AnnotationBbox(imagebox, (x_icon, y), frameon=False, xycoords='data', boxcoords="offset points", pad=0)
                    ax.add_artist(ab)
                except:
                    # If image fails to load, use a default marker
                    ax.plot(x_icon, y, marker='o', color=color, markersize=8)
            else:
                # If icon image not found, use a default marker
                ax.plot(x_icon, y, marker='o', color=color, markersize=8)

            # Wrap text to fit within a certain width
            wrapped_text = textwrap.fill(item, width=20)

            # Add text
            if side == 'pros':
                ha = 'left'
            else:
                ha = 'right'

            ax.text(x_text, y, wrapped_text, ha=ha, va='center', fontsize=12,
                    fontproperties=text_font, color='black',
                    bbox=dict(facecolor='none', edgecolor='none', pad=0))

    # Add Pros and Cons items
    add_items(pros, 'pros')
    add_items(cons, 'cons')

    # Adjust plot limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('pros_cons.jpg', dpi=300, bbox_inches='tight')
    

# Visualization for Circular Diagram

def plot_circular_diagram(circular_theme, circular_sub_elements):
    
    for i in range(len(circular_sub_elements)):
        circular_sub_elements[i] += ', icon_'+str(i)+'.png'
        
    # Convert each string to a tuple (segment_name, icon_name)
    for i in range(len(circular_sub_elements)):
        # Split the string by ', ' and convert to tuple
        circular_sub_elements[i] = tuple(circular_sub_elements[i].split(', '))


    print(circular_sub_elements)

    # Determine the number of segments
    num_segments = len(circular_sub_elements)
    print(f"Number of segments: {num_segments}")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Load custom font if available
    try:
        font_path = 'PatrickHand-Regular.ttf'  # Ensure this font file is in your working directory
        title_font = FontProperties(fname=font_path, size=20, weight='bold')
        label_font = FontProperties(fname=font_path, size=12)
    except:
        # Fallback to default font
        print("Custom font not found. Using default font.")
        title_font = {'size': 20, 'weight': 'bold'}
        label_font = {'size': 12}
    
    # Define pastel color palette
    pastel_colors = plt.cm.Pastel1(np.linspace(0, 1, num_segments))
    print(f"Assigned pastel colors: {pastel_colors}")
    
    # Calculate angle for each segment
    angles = np.linspace(0, 360, num_segments + 1)
    print(f"Segment angles: {angles}")
    
    # Draw each segment as a Wedge
    for i, segment in enumerate(circular_sub_elements):
        print(f"Processing segment {i+1}: {segment}")
        
        # Handle different data structures
        if isinstance(segment, tuple):
            if len(segment) != 2:
                raise ValueError(f"Invalid tuple format at index {i}: {segment}. Expected (segment_name, icon_name).")
            segment_name, icon_name = segment
            print(f"Segment name: {segment_name}, Icon name: {icon_name}")
        elif isinstance(segment, str):
            segment_name = segment
            icon_name = None
            print(f"Segment name: {segment_name}, No icon.")
        else:
            raise ValueError(f"Invalid segment format at index {i}: {segment}. Expected a string or a tuple of (str, str).")
        
        # Define the Wedge
        wedge = Wedge(center=(0, 0),
                      r=1,
                      theta1=angles[i],
                      theta2=angles[i + 1],
                      facecolor=pastel_colors[i],
                      edgecolor='white',
                      linewidth=2)
        ax.add_patch(wedge)
        print(f"Added Wedge for segment {i+1}: {segment_name}")
        
        # Calculate the angle for placing text and icon
        theta = (angles[i] + angles[i + 1]) / 2
        theta_rad = np.deg2rad(theta)
        print(f"Segment {i+1} central angle (degrees): {theta}, (radians): {theta_rad}")
        
        # Position for the text (70% radius)
        text_radius = 0.7
        text_x = text_radius * np.cos(theta_rad)
        text_y = text_radius * np.sin(theta_rad)
        print(f"Text position for segment {i+1}: ({text_x}, {text_y})")
        
        # Wrap the text to fit within the segment
        wrapped_text = textwrap.fill(segment_name, width=15)
        print(f"Wrapped text for segment {i+1}: {wrapped_text}")
        
        # Add the text
        ax.text(text_x, text_y, wrapped_text, ha='center', va='center',
                fontsize=12, fontproperties=label_font, wrap=True)
        print(f"Added text for segment {i+1}")
        
        # Position for the icon (50% radius)
        icon_radius = 0.5
        icon_x = icon_radius * np.cos(theta_rad)
        icon_y = icon_radius * np.sin(theta_rad)
        print(f"Icon position for segment {i+1}: ({icon_x}, {icon_y})")
        
        # Add the icon if available
        if icon_name:
            icon_path = icon_name  # Assuming icon_name is the path
            print(f"Attempting to load icon for segment {i+1}: {icon_path}")
            if os.path.exists(icon_path):
                try:
                    img = mpimg.imread(icon_path)
                    imagebox = OffsetImage(img, zoom=0.1)  # Adjust zoom as needed
                    ab = AnnotationBbox(imagebox, (icon_x, icon_y),
                                        frameon=False, 
                                        xycoords='data',
                                        boxcoords="offset points",
                                        pad=0)
                    ax.add_artist(ab)
                    print(f"Added icon for segment {i+1}: {icon_path}")
                except Exception as e:
                    print(f"Error loading icon '{icon_path}': {e}")
                    # Optionally, add a default marker if icon loading fails
                    ax.plot(icon_x, icon_y, marker='o', color='gray', markersize=5)
                    print(f"Added default marker for segment {i+1} due to icon load failure")
            else:
                # Optionally, add a default marker if icon not found
                ax.plot(icon_x, icon_y, marker='o', color='gray', markersize=5)
                print(f"Icon file not found for segment {i+1}: {icon_path}")
        else:
            print(f"No icon specified for segment {i+1}.")
    
    # Add central icon or text
    central_icon_path = 'central_icon.png'  # Ensure you have a central icon
    if os.path.exists(central_icon_path):
        try:
            img = mpimg.imread(central_icon_path)
            imagebox = OffsetImage(img, zoom=0.2)  # Adjust zoom as needed
            ab = AnnotationBbox(imagebox, (0, 0),
                                frameon=False, 
                                xycoords='data',
                                boxcoords="offset points",
                                pad=0)
            ax.add_artist(ab)
            print(f"Added central icon: {central_icon_path}")
        except Exception as e:
            print(f"Error loading central icon '{central_icon_path}': {e}")
            # If image loading fails, add central text
            ax.text(0, 0, "Clinical AI\nMarketplace", ha='center', va='center',
                    fontsize=14, fontproperties=title_font, color='black')
            print("Added central text instead of icon.")
    else:
        # If central icon not found, add central text
        ax.text(0, 0, "Clinical AI\nMarketplace", ha='center', va='center',
                fontsize=14, fontproperties=title_font, color='black')
        print("Central icon not found. Added central text.")
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal')
    
    # Remove axes
    plt.axis('off')
    
    # Add title
    plt.title(circular_theme, fontsize=20, fontproperties=title_font, y=1.05)
    
    plt.tight_layout()
    plt.savefig('circular_diagram_with_icons.jpg', dpi=300, bbox_inches='tight')
    




# Visualization for Decision Tree
def plot_decision_tree(decision_tree_data):
    """
    Plots a decision tree with a horizontal layout, icons, color-coded branches, and readable text.

    Parameters:
    - decision_tree_data: List of tuples representing parent-child relationships.
      Example:
        [
            ('Start', 'Option A'),
            ('Start', 'Option B'),
            ('Option A', 'Result A1'),
            ('Option A', 'Result A2'),
            ('Option B', 'Result B1'),
            ('Option B', 'Result B2'),
        ]
    """
    if not decision_tree_data:
        print("No decision tree data available.")
        return

    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(decision_tree_data)

    # Assign levels to nodes using BFS
    root = decision_tree_data[0][0]  # Assuming the first element is the root
    levels = {root: 0}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for neighbor in G.successors(current):
            if neighbor not in levels:
                levels[neighbor] = levels[current] + 1
                queue.append(neighbor)

    # Assign positions to nodes manually for a horizontal layout
    # Nodes at the same level are spaced evenly vertically
    pos = {}
    level_nodes = {}
    for node, level in levels.items():
        level_nodes.setdefault(level, []).append(node)

    # Determine spacing
    max_level = max(level_nodes.keys())
    fig_width = max_level * 10  # Adjust as needed
    fig_height = max(len(nodes) for nodes in level_nodes.values()) * 4  # Adjust as needed

    # Calculate positions with adequate spacing
    for level, nodes in level_nodes.items():
        num_nodes = len(nodes)
        # Avoid division by zero
        if num_nodes > 1:
            y_gap = 4  # Gap between nodes vertically
            y_start = (num_nodes - 1) * y_gap / 2
            for i, node in enumerate(nodes):
                pos[node] = (level * 10, y_start - i * y_gap)
        else:
            pos[nodes[0]] = (level * 10, 0)

    # Initialize plot
    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()
    ax.set_xlim(-5, (max_level + 1) * 10)
    ax.set_ylim(-fig_height / 2, fig_height / 2)
    ax.axis('off')

    # Define color palette for branches
    colors = plt.cm.tab20.colors  # A colormap with enough distinct colors
    color_map = {}
    for i, edge in enumerate(G.edges()):
        color = colors[i % len(colors)]
        color_map[edge] = color

    # Draw edges with colors
    for edge in G.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        ax.plot([x1, x2], [y1, y2], color=color_map[edge], linewidth=2, zorder=1)

    # Draw nodes
    for node in G.nodes():
        x, y = pos[node]
        # Draw a circle for the node
        circle = Circle((x, y), 1.5, color='white', ec='black', lw=2, zorder=2)
        ax.add_patch(circle)

        # Add icon if available
        icon_path = f'icon_{node}.png'  # Ensure icons are named accordingly
        if os.path.exists(icon_path):
            try:
                img = mpimg.imread(icon_path)
                imagebox = OffsetImage(img, zoom=0.6)  # Adjust zoom as needed
                ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=3)
                ax.add_artist(ab)
            except:
                # If image fails to load, skip adding the icon
                pass

        # Add text label next to the node
        # Wrap text to fit within a certain width
        wrapped_text = textwrap.fill(node, width=15)

        # Determine text position based on node's level to prevent overlap
        if pos[node][0] == 0:
            # Root node, place text to the right
            text_x = x + 2.0
            ha = 'left'
        else:
            # Other nodes, place text to the right
            text_x = x + 2.0
            ha = 'left'

        # Add text with wrapping and adaptive font size
        ax.text(text_x, y, wrapped_text, ha=ha, va='center', fontsize=12,
                fontweight='bold', wrap=True, zorder=4)

    # Add labels above the edges to prevent overlapping
    for edge in G.edges():
        parent, child = edge
        x1, y1 = pos[parent]
        x2, y2 = pos[child]
        # Calculate midpoint
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        # Offset the label above the line
        dx = x2 - x1
        dy = y2 - y1
        distance = (dx**2 + dy**2)**0.5
        if distance == 0:
            offset_x, offset_y = 0, 0
        else:
            offset_x, offset_y = -dy / distance, dx / distance  # Perpendicular direction

        # Define annotation text (optional)
        # For example, if you have labels for branches, include them here
        # Here, we'll use the child node name as an example
        annotation_text = ""  # Empty if no additional labels are needed

        if annotation_text:
            ax.text(xm + offset_x * 1.5, ym + offset_y * 1.5, annotation_text,
                    ha='center', va='center', fontsize=10, color='black', zorder=5)

    # Add title
    try:
        title_font = FontProperties(fname='PatrickHand-Regular.ttf', size=24)
        plt.title("Decision Tree", fontproperties=title_font, fontsize=24, y=1.05)
    except:
        # Fallback to default font if custom font is not found
        plt.title("Decision Tree", fontsize=24, fontweight='bold', y=1.05)

    plt.tight_layout()
    plt.savefig('decision_tree.jpg', dpi=300, bbox_inches='tight')
    plt.show()


# Visualization for Timeline (Already Beautiful)
def plot_timeline(timeline_data):
    if not timeline_data:
        print("No timeline data available.")
        return

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_facecolor(COLOR_PALETTE['background'])

    # Define colors
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#8BC34A', '#FFC107', '#00BCD4', '#E91E63']
    while len(colors) < len(timeline_data):
        colors += colors  # Repeat colors if necessary

    # Load a playful font (ensure the font file is accessible)
    title_font = FontProperties(fname='PatrickHand-Regular.ttf', size=18)
    event_font = FontProperties(fname='PatrickHand-Regular.ttf', size=12)
    year_font = FontProperties(fname='PatrickHand-Regular.ttf', size=14, weight='bold')

    # Plot vertical line for the timeline
    ax.plot([0, 0], [-0.5, len(timeline_data) - 0.5], color='gray', lw=2, zorder=1)

    # Plot each event
    for i, (year, event) in enumerate(timeline_data):
        color = colors[i % len(colors)]  # Cycle through colors if necessary

        # Alternate side placement
        side = -1 if i % 2 == 0 else 1

        # Positioning
        x_text = side * 1.0
        x_marker = 0
        y_pos = len(timeline_data) - i - 1  # Reverse y-axis to have the earliest event at the top

        # Draw dotted line from marker to text
        ax.plot([x_marker, x_text], [y_pos, y_pos], color='gray', lw=1, linestyle='dotted')

        # Draw the icon marker (circle with icon)
        circle = Circle((x_marker, y_pos), 0.2, color=color, ec='black', lw=1, alpha=0.9, zorder=2)
        ax.add_artist(circle)

        # Load and place the icon
        icon_path = f'icon_{i}.png'  # Ensure you have icons named accordingly
        if os.path.exists(icon_path):
            img = mpimg.imread(icon_path)
            imagebox = OffsetImage(img, zoom=0.3)
            ab = AnnotationBbox(imagebox, (x_marker, y_pos), frameon=False)
            ax.add_artist(ab)
        else:
            # If icon image not found, use a placeholder (e.g., a simple marker)
            ax.plot(x_marker, y_pos, marker='o', markersize=12, color='white', markeredgecolor='black', zorder=3)

        # Place the year text
        ax.text(x_text, y_pos + 0.3, year, va='center', ha='center', fontsize=14, fontweight='bold',
                color=color, fontproperties=year_font)

        # Place the event description
        ax.text(x_text, y_pos - 0.3, event, va='center', ha='center', fontsize=12, color='black',
                fontproperties=event_font, wrap=True, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

    # Set limits, remove axes
    ax.set_xlim([-2, 2])
    ax.set_ylim([-1, len(timeline_data)])
    ax.axis('off')

    # Add the title at the top
    plt.title("Dandelion Health's AI Journey: From Data Library to Clinical AI Marketplace",
              fontsize=20, fontproperties=title_font, pad=30, color=COLOR_PALETTE['title'], wrap=True)

    # Add a downward arrow at the bottom to indicate continuation
    arrow = FancyArrowPatch((0, -0.5), (0, -1), mutation_scale=20, color='gray', lw=2, arrowstyle='-|>')
    ax.add_artist(arrow)

    plt.tight_layout()
    plt.savefig('timeline_visualization.jpg', dpi=300, bbox_inches='tight')
    

# Function to generate visualizations
def generate_visualizations(article_text=None, use_existing_response=True):
    if not use_existing_response or not os.path.exists('openai_response.txt'):
        print("Fetching new data from OpenAI...")
        response = ask_llm(article_text)
    else:
        print("Using existing OpenAI response from file...")
        with open('openai_response.txt', 'r') as f:
            response = f.read()

    # Parse the response into structured data
    data = parse_llm_response(response)
    
    # Generate all visualizations
    plot_funnel(data["funnel"])
    plot_timeline(data["timeline"])
    plot_pros_cons(data["pros"], data["cons"])
    plot_circular_diagram(data["circular_theme"], data["circular_sub_elements"])
    plot_decision_tree(data["decision_tree"])

# Function to read article from a file
def read_article_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Main function execution
if __name__ == "__main__":
    # Replace this with the file path to your article text
    file_path = './article.txt'

    # Set to True if you want to use an existing response file, or False to fetch a new one
    use_existing_response = True

    # Read the article text from the file
    article_text = read_article_from_file(file_path)

    # Run the visualization generation
    generate_visualizations(article_text, use_existing_response)