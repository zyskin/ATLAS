(* Plot every valid extracted M3 pixel in the Lucey plane.              *)
(* DensityHistogram uses all retained pixels; it does not subsample.    *)

(* ---------------- configuration ---------------- *)
stage1Directory = "/media/maxim/Elements/grants/2024/INSRU/pgdal/m3_stage1_retest_2026-10-01/all_memmap";
prefix = "M3G20090107T011405";
outputDirectory = "/media/maxim/Elements/grants/2024/INSRU/pgdal/m3_lucey_plots_mathematica";
definitions = {"direct", "kumar_mean"};
reflectanceMaximum = 2.0;
ratioMaximum = 5.0;
gridSize = 260;
(* ------------------------------------------------- *)

luceyOrigin = {0.08, 1.19};
kumarOrigin = {0.08, 1.18};

bandFile[band_Integer] := FileNameJoin[{stage1Directory,
    prefix <> "_band_" <> ToString[band] <> ".tsv"}];
readBand[band_Integer] := Developer`ToPackedArray[N@Import[bandFile[band], "TSV"]];

coordinates[definition_String] := Module[
  {lowBands, highBands, low, high, ratio, pairs},
  If[definition == "direct",
    lowBands = {9}; highBands = {19},
    lowBands = {8, 9, 10}; highBands = {18, 19, 20}
  ];
  low = Mean[readBand /@ lowBands];
  high = Mean[readBand /@ highBands];
  ratio = high/low;
  pairs = Transpose[{Flatten[low], Flatten[ratio]}];
  Select[pairs,
    VectorQ[#, NumericQ] && 0 < #[[1]] <= reflectanceMaximum &&
      0 <= #[[2]] <= ratioMaximum &]
];

CreateDirectory[outputDirectory, CreateIntermediateDirectories -> True];

Do[
  points = coordinates[definition];
  density = DensityHistogram[
    points, {gridSize, gridSize}, "Count",
    ColorFunction -> "Viridis",
    PlotLegends -> Automatic,
    FrameLabel -> {Style[Subscript[R, 750], 13],
      Style[Subscript[R, 950]/Subscript[R, 750], 13]},
    PlotLabel -> Style[prefix <> " — " <> definition <>
      "; all valid pixels; n = " <> ToString[Length[points]], 15],
    ImageSize -> 1000
  ];
  origins = Graphics[{
     {Red, PointSize[0.020], Point[luceyOrigin],
       Text[Style["Lucey origin", 13, Red], luceyOrigin, {-1.1, -1.2}]},
     {Darker[Orange], PointSize[0.020], Point[kumarOrigin],
       Text[Style["Kumar origin", 13, Darker[Orange]], kumarOrigin, {1.1, 1.2}]}
  }];
  output = FileNameJoin[{outputDirectory,
     prefix <> "_lucey_plane_" <> definition <> ".png"}];
  Export[output, Show[density, origins], ImageResolution -> 220];
  Print[definition, ": ", Length[points], " valid pixels -> ", output],
  {definition, definitions}
];

