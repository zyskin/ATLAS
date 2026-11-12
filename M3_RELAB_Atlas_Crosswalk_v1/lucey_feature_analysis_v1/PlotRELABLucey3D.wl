(* Plot RELAB spectra in {R750, R950/R750, FeO} coordinates.            *)
(* Edit only the configuration block, then evaluate the complete file. *)

(* ---------------- configuration ---------------- *)
projectRoot = "/media/maxim/Elements/grants/2024/INSRU/pgdal/M3_RELAB_Atlas_Crosswalk_v1";
inputCSV = FileNameJoin[{projectRoot, "data", "ml_combined_v2",
   "m3_feo_isfeo_ground_truth_v2.csv"}];
outputPNG = FileNameJoin[{projectRoot, "data", "lucey_feature_analysis_v1",
   "relab_lucey_3d_combined.png"}];
targetColumn = "feo_wt_percent_mean";
(* For strict specimen-level FeO use "feo_direct_wt_percent_mean". *)
(* ------------------------------------------------- *)

luceyOrigin = {0.08, 1.19};
kumarOrigin = {0.08, 1.18};

raw = Import[inputCSV, "CSV"];
header = First[raw];
rows = Rest[raw];
column[name_] := First@FirstPosition[header, name];
toNumber[x_] := Quiet@Check[N@ToExpression[x], Indeterminate];
finiteNumberQ[x_] := NumberQ[x] && x =!= Indeterminate && -Infinity < x < Infinity;

xColumn = column["r750_direct_b09"];
yColumn = column["r950_over_r750_direct"];
zColumn = column[targetColumn];

points = Cases[
   rows,
   row_ :> With[{x = toNumber[row[[xColumn]]],
      y = toNumber[row[[yColumn]]], z = toNumber[row[[zColumn]]]},
     If[finiteNumberQ[x] && finiteNumberQ[y] && finiteNumberQ[z] && x > 0 && y > 0,
       {x, y, z}, Nothing]
   ]
];

If[Length[points] == 0, Print["No finite points found."]; Abort[]];
zRange = MinMax[points[[All, 3]]];
zFloor = Min[0., zRange[[1]]];

dataPlot = ListPointPlot3D[
   points,
   PlotStyle -> Directive[PointSize[0.010], Opacity[0.72]],
   ColorFunction -> Function[{x, y, z}, ColorData["Viridis"][Rescale[z, zRange]]],
   ColorFunctionScaling -> False,
   AxesLabel -> {Style[Subscript[R, 750], 13],
     Style[Subscript[R, 950]/Subscript[R, 750], 13], Style["FeO (wt.%)", 13]},
   PlotRange -> All,
   BoxRatios -> {1.1, 1., 0.8},
   PlotLegends -> BarLegend[{"Viridis", zRange}, LegendLabel -> "FeO (wt.%)"],
   ImageSize -> 1000
];

originPlot = Graphics3D[{
    {Red, Thick, Dashed,
      Line[{{luceyOrigin[[1]], luceyOrigin[[2]], zFloor},
        {luceyOrigin[[1]], luceyOrigin[[2]], zRange[[2]]}}],
      PointSize[0.022], Point[Append[luceyOrigin, zFloor]],
      Text[Style["Lucey origin", 13, Red], Append[luceyOrigin, zFloor], {-1.2, -1.2}]},
    {Darker[Orange], Thick, Dashed,
      Line[{{kumarOrigin[[1]], kumarOrigin[[2]], zFloor},
        {kumarOrigin[[1]], kumarOrigin[[2]], zRange[[2]]}}],
      PointSize[0.022], Point[Append[kumarOrigin, zFloor]],
      Text[Style["Kumar origin", 13, Darker[Orange]],
        Append[kumarOrigin, zFloor], {1.2, 1.2}]}
}];

plot = Show[dataPlot, originPlot,
   PlotLabel -> Style[
     "RELAB spectra on Lucey plane; height = " <> targetColumn <>
       "; n = " <> ToString[Length[points]], 15]
];
CreateDirectory[DirectoryName[outputPNG], CreateIntermediateDirectories -> True];
Export[outputPNG, plot, ImageResolution -> 220];
Print["Points plotted: ", Length[points]];
Print["Output: ", outputPNG];

