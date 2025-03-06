from csv import DictReader
from matplotlib import pyplot
from pathlib import Path

def filter_split(filter_name: str) -> tuple[str, int]:
    name_part, number_part = filter_name.rsplit('_', 1)
    return name_part, int(number_part)

def containment_line_graph(
    data_file: str,
    dataset_name: str,
    containment_filters: list[str],
    filter_group: list[str],
    stat: str,
    title: str,
    center: str = 'mean',
    max_strength: int = 10):
  total_set = containment_filters + filter_group
  x = {}
  y = {}
  y1 = {}
  y2 = {}
  for filter_name in total_set:
    x[filter_name] = [i for i in range(max_strength)]
    y1[filter_name] = [None for _ in range(max_strength)]
    y2[filter_name] = [None for _ in range(max_strength)]
    y[filter_name] = [None for _ in range(max_strength)]
  with open(Path(data_file).resolve(), 'r') as f:
    reader = DictReader(f)
    for row in reader:
      if row['dataset_name'] != dataset_name or row['filter_name'] == 'all_filters':
        continue
      filter_name, strength = filter_split(row['filter_name'])
      if filter_name not in total_set:
        continue
      y[filter_name][strength - 1] = float(row[f'{stat}_{center}'])
      if filter_name in containment_filters:
        y1[filter_name][strength - 1] = float(row[f'{stat}_{center}']) - float(row[f'{stat}_std'])
        y2[filter_name][strength - 1] = float(row[f'{stat}_{center}']) + float(row[f'{stat}_std'])
  for filter_name in containment_filters:
    pyplot.fill_between(x[filter_name], y1[filter_name], y2[filter_name], alpha=0.2)
  for filter_name in total_set:
    pyplot.plot(x[filter_name], y[filter_name])
  pyplot.title(title)
  pyplot.show()

edge_group = ['thin_edge_enhancement', 'thick_edge_enhancement', 'kuwahara_filter']
color_group = ['color_quantization', 'hue_shift', 'saturate', 'desaturate', 'contrast', 'decontrast' 'invert', 'darken_shadows', 'vignette']
frequency_group = ['gaussian_blur', 'gaussian_high_pass', 'horizontal_blur', 'vertical_blur', 'focus_blur', 'bloom_filter']
digital_group = ['pixelate', 'row_shift', 'screen_door_effect', 'chromatic_aberration']

for dataset in ['general', 'MIT1003', 'MIT300']:
  containment_line_graph(
    'aggregate_experiment_data.csv',
    dataset,
    ['gaussian_noise', 'perlin_noise'],
    edge_group,
    'divergence_per_difference',
    f"Divergence Per Difference for Edge Group in {dataset}"
  )

for group in [[], edge_group, color_group, frequency_group, digital_group]:
  group_name = ''
  if group == []:
    group_name = 'Noise Distributions Only'
  elif group == edge_group:
    group_name = 'Edge Group'
  elif group == color_group:
    group_name = 'Color Group'
  elif group == frequency_group:
    group_name = 'Frequency Group'
  elif group == digital_group:
    group_name = 'Digital Group'

  containment_line_graph(
    'aggregate_experiment_data.csv',
    'general',
    ['gaussian_noise', 'perlin_noise'],
    group,
    'divergence_per_difference',
    f"Divergence Per Difference for {group_name}"
  )

  containment_line_graph(
    'aggregate_experiment_data.csv',
    'general',
    ['gaussian_noise', 'perlin_noise'],
    group,
    'information_asymmetry_per_difference',
    f"Information Asymmetry Per Difference for {group_name}"
  )
