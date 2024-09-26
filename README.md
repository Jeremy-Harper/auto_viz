# OpenAI LLM Visualization Tool

This project is a Python-based tool that leverages a LLM to analyze articles and generate multiple types of visualizations. It creates Funnel Diagrams, Timelines, Pros and Cons tables, Circular Diagrams, and Decision Trees based on the structured data extracted from an article.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Fetching Data from LLM](#fetching-data-from-LLM)
  - [Generating Visualizations](#generating-visualizations)
- [Benefits](#benefits)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install and run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/openai-llm-visualization.git
   cd openai-llm-visualization

Set up a virtual environment and install the required dependencies:

bash

python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt

Make sure you have a valid OpenAI API key. Set it as an environment variable:

bash

    export OPENAI_API_KEY='your-openai-api-key'

    (Optional) To use custom fonts or icons, ensure the required font files (e.g., PatrickHand-Regular.ttf) and icons are placed in the working directory.

Usage
Fetching Data from OpenAI

The tool interacts with OpenAI's API to fetch data from a given article using the ask_llm function. The fetched data is then parsed and used to generate structured data for visualizations such as funnels, timelines, pros/cons lists, circular diagrams, and decision trees.
Example:

    Prepare your article text in a file (e.g., article.txt).
    Call the main function, generate_visualizations, to analyze the text and generate visualizations:

    bash

    python main.py

The tool will create the following visualizations:

    Funnel Diagram
    Timeline
    Pros and Cons Table
    Circular Diagram
    Decision Tree

The generated visualizations will be saved as image files (e.g., funnel_diagram.jpg, pros_cons.jpg) in the working directory.
Detailed Functionality
1. Funnel Diagram

    Visualizes stages of a process (e.g., sales funnel) with customizable icons for each stage.

2. Timeline

    Extracts milestones from the article and displays them along a vertical timeline.

3. Pros and Cons Table

    Lists pros and cons side by side, visually separated by a central "VS" marker.

4. Circular Diagram

    Visualizes a core theme surrounded by sub-elements, with support for icons.

5. Decision Tree

    Shows decisions and their possible outcomes in a clear tree structure with color-coded branches.

See the JPGS for examples
1. Decision Tree Diagram

![Decision Tree Diagram](https://github.com/Jeremy-Harper/auto_viz/blob/main/decision_tree.jpg)

2. Circular Diagram

![Circular Diagram](https://github.com/Jeremy-Harper/auto_viz/blob/main/circular_diagram_with_icons.jpg)

3. Pros and Cons

![Pros and Cons](https://github.com/Jeremy-Harper/auto_viz/blob/main/pros_cons.jpg)

4. Timeline Visualization
   
![Timeline Visualization](https://github.com/Jeremy-Harper/auto_viz/blob/main/timeline_visualization.jpg)

6. Funnel Diagram

![Funnel Diagram](https://github.com/Jeremy-Harper/auto_viz/blob/main/funnel_diagram.jpg)

Key Benefits

    Automated Visualization: Automatically generates multiple types of visual diagrams from an article with minimal manual input.
    Clear Visual Communication: The tool produces professional and clean visuals, suitable for presentations and reports.
    Highly Customizable: The visual styles (e.g., colors, fonts) can be easily customized by adjusting the palette or adding custom icons and fonts.
    Efficient Data Parsing: The integration with OpenAI ensures that the data extraction and structuring is efficient and capable of handling complex articles.

Customization

    Seaborn Styling: Seaborn is used to set up global styles, providing professional-looking plots with minimal configuration.
    Font Customization: You can specify custom fonts (e.g., PatrickHand-Regular.ttf) to give your visualizations a unique touch.
    Color Palette: Modify the COLOR_PALETTE dictionary to adjust the colors used across all visualizations.

Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request with your changes.
