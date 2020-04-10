<?php

echo "Started at ".date("H:m:s");

$in_raw = file_get_contents('train.csv');
$in = explode("\n",$in_raw);

//helper function to get the Default Save Value
function DefSV($def,$obj,$key) {
	if (isset($obj[$key])) {
		return $obj[$key];
	} else {
		return $def;
	}
}

//luckily, the data is clean enough for doing dirty csv processing
$keyed = [];
$colmap = explode(",",$in[0]);
for($i = 1; $i < count($in); $i++) {
	$line = explode(",",$in[$i]);
	if (count($line) == 1) continue;
	
	$row = [];
	foreach ($colmap as $c=>$col) {
		$row[$col] = $line[$c];
	}
	$keyed[] = $row;
}
//echo "<pre>".var_export($keyed,true)."</pre>";

//Now, find all values, for enumeration, and/or column layout
$alls = [];
$splits = [];
foreach ($keyed as $i=>$row) { foreach($row as $col=>$val) {
	$alls[$col][$val] = $val;
	
	if ($col == "Breed") {
		$ex = explode("/",str_replace(" Mix","",$val));
		foreach($ex as $exv) { $splits[$col.'-'.$exv] = []; }
	} else if ($col == "Color") {
		$ex = explode("/",$val);
		foreach($ex as $exv) { $splits[$col.'-'.$exv] = []; }
	} else if ($col == "AnimalType") {
		$splits[$col.'-'.$val] = [];
	} else if ($col == "SexuponOutcome") {
		// ignore, done manually below
	} else {
		$splits[$col][$val] = $val;
	}
}}
//Add a few more columns to the extended feature version
$splits["SexuponOutcome-Female"] = [];
$splits["SexuponOutcome-Male"]   = [];
$splits["SexuponOutcome-Sterilized"] = [];
$splits["SexuponOutcome-Unknown"] = [];
$splits["Breed-Mix"] = [];
$splits["Color-Mix"] = [];
for($i = 1; $i <= 12; $i++) {
	$splits["DateTime-Month{$i}"] = [];
}
for ($i = 1; $i <= 7; $i++) {
	$splits["DateTime-DOW{$i}"] = [];
}
$splits["DateTime-Morning"] = [];
$splits["DateTime-Afternoon"] = [];
$splits["DateTime-Night"] = [];

//Get rid of columns we won't be keeping
unset($alls['AnimalID']);
unset($splits['AnimalID']);
unset($alls['OutcomeSubtype']);
unset($splits['OutcomeSubtype']);

ksort($alls);
ksort($splits);

//echo "<pre>".var_export($alls,true)."</pre>";
//echo "<pre>".var_export($splits,true)."</pre>";

//Go through each row, and each data cell, and transform it.
foreach ($keyed as $i=>&$row) { foreach($row as $col=>$val) {
	
	if ($col == "AnimalID") {
		unset($row[$col]); //this doesn't get used as an input, so would require
	}
	if ($col == "Name") {
		$row[$col] = ($val ? 1:0);
	}
	if ($col == "DateTime") {
		$row[$col] = strtotime($val);
		$ex = explode(" ",$val);
		$exh = explode(":",$ex[1]);  $hour = intval($exh[0]);
		$exm = explode("-",$ex[0]);  $mon  = intval($exm[1]);
		$row[$col.'-Month'.$mon] = 1;
		$row[$col.'-DOW'.date('N')] = 1;
		if ($hour >= 7 && $hour < 12) {
			$row[$col.'-Morning'] = 1;
		} else if ($hour >= 12 && $hour <= 19) {
			$row[$col.'-Afternoon'] = 1;
		} else { // $hour < 7 || $hour > 19
			$row[$col.'-Night'] = 1;
		}
	}
	if ($col == "AgeuponOutcome") {
		$ex = explode(" ",$val);
		$ex[0] = intval($ex[0]);
		if ($val == "") {
			$row[$col] = 7.77; //unknown value, distinct though
		} else if ($ex[1] == "years" || $ex[1] == "year") {
			$row[$col] = $ex[0];
		} else if ($ex[1] == "months" || $ex[1] == "month") {
			$row[$col] = $ex[0]/12;
		} else if ($ex[1] == "weeks" || $ex[1] == "week") {
			$row[$col] = $ex[0]/52;
		} else if ($ex[1] == "days" || $ex[1] == "day") {
			$row[$col] = $ex[0]/365;
		}
	}
	if ($col == "OutcomeType") {
		//no modification, final combiner will put this last
	}
	if ($col == "OutcomeSubtype") {
		unset($row[$col]); //this is not in the "test.csv" so we can't use it for the challenge
	}
	if ($col == "AnimalType") {
		$row[$col] = array_search($val,array_keys($alls[$col]));
		$row[$col.'-'.$val] = 1;
	}
	if ($col == "SexuponOutcome") {
		if ($val == "") { $val = "Unknown"; } //one blank to fill in
		$row[$col] = array_search($val,array_keys($alls[$col]));
		$row[$col.'-Unknown'] = ($val=="Unknown"?1:0);
		$row[$col.'-Sterilized'] = (strpos($val,"Neutered") !== strpos($val,'Spayed') ? 1:0); //both "false" if Intact and not Unkown
		$row[$col.'-Male'] = (strpos($val,"Male") !== false ? 1:0);
		$row[$col.'-Female'] = (strpos($val,"Female") !== false ? 1:0);
	}
	if ($col == "Breed") {
		$row[$col] = array_search($val,array_keys($alls[$col]));
		$mix = (strpos($val," Mix") !== false ? 1:0);
		$ex = explode("/",str_replace(" Mix","",$val));
		$mix = (count($ex) > 1 ? 1:$mix);
		foreach($ex as $exv) { $row[$col.'-'.$exv] = 1; }
		$row[$col.'-Mix'] = $mix;
	}
	if ($col == "Color") {
		$row[$col] = array_search($val,array_keys($alls[$col]));
		$ex = explode("/",$val);
		$mix = (count($ex) > 1 ? 1:0);
		foreach($ex as $exv) { $row[$col.'-'.$exv] = 1; }
		$row[$col.'-Mix'] = $mix;
	}
}} unset($row);

//echo "<pre>".var_export($keyed,true)."</pre>";

//function to convert the definition array and data array into CSV ready lines
function Transform($define, $keyed, $header) {
	$conv = "";
	if ($header) {
		$head = [];
		foreach($define as $col=>$vals) {
			if ($col == "OutcomeType") continue;
			$head[] = $col;
		}
		$head[] = "OutcomeType";
		$conv .= implode(",",$head)."\n";
	}
	foreach ($keyed as $i=>$row) {
		$line = ""; $d = "";
		foreach($define as $col=>$vals) {
			if ($col == "OutcomeType") continue;
			
			$line .= $d.DefSV(0,$row,$col);
			$d = ",";
		}
		$line .= ",".DefSV("",$row,"OutcomeType");
		
		$conv .= $line."\n";
	}
	return $conv;
}

	
//write the various versions to the files
file_put_contents("train_conversion_lowfeat_withheader.csv",Transform($alls,  $keyed, true));
file_put_contents("train_conversion_lowfeat_sansheader.csv",Transform($alls,  $keyed, false));
file_put_contents("train_conversion_extfeat_withheader.csv",Transform($splits,$keyed, true));
file_put_contents("train_conversion_extfeat_sansheader.csv",Transform($splits,$keyed, false));


echo "<br/>Finished at ".date("H:m:s");