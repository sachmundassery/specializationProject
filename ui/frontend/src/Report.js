import React, { useEffect, useState, useRef } from "react";
import { Line } from "react-chartjs-2";

const Report = () => {
	const [csvData, setCsvData] = useState([]);
	const [completeCsvData, setCompleteCsvData] = useState([]);
	const [incomingCsvData, setIncomingCsvData] = useState();
	const [searchTerm, setSearchTerm] = useState("");
	const inputElement = useRef("");
	const [searchResult, setSearchResult] = useState([]);

	const [showChartStatus, setShowChartStatus] = useState(false);
	const [currentPage, setCurrentPage] = useState(1);
	const [itemPerPage, setItemPerPage] = useState(6);
	const [pages, setPages] = useState([]);

	useEffect(async () => {
		const formData = new FormData();
		formData.append("type", "fetchCsv");

		const response = await fetch("http://127.0.0.1:8000/company", {
			method: "POST",
			body: formData,
		});
		const result = await response.json();
		setIncomingCsvData(() => {
			return result;
		});

		// to get number of Pages
		for (
			let i = 1;
			i <=
			Math.ceil(Object.keys(result["csvDataFrameArray"]).length / itemPerPage);
			i++
		) {
			setPages((pages) => [...pages, i]);
		}
	}, []);

	const fetchFullCsvData = () => {
		var fullCsvData = {};
		var fullCsvDataDictArray = [];

		for (
			var i = 0;
			i < Object.keys(incomingCsvData["csvDataFrameArray"]).length;
			i++
		) {
			let tempDict = {};
			tempDict[Object.keys(incomingCsvData["csvDataFrameArray"])[i]] =
				Object.values(incomingCsvData["csvDataFrameArray"])[i];
			fullCsvDataDictArray.push(tempDict);
		}

		for (
			var i = 0;
			i < Object.keys(incomingCsvData["csvDataFrameArray"]).length;
			i++
		) {
			var sales = [];
			var customers = [];
			var promo = [];
			var dates = [];

			for (
				var j = 0;
				j < Object.values(Object.values(fullCsvDataDictArray[i])[0]).length;
				j++
			) {
				sales.push(
					Object.values(Object.values(fullCsvDataDictArray[i])[0])[j]["Sales"]
				);
				customers.push(
					Object.values(Object.values(fullCsvDataDictArray[i])[0])[j][
						"Customers"
					]
				);
				promo.push(
					Object.values(Object.values(fullCsvDataDictArray[i])[0])[j]["Promo"]
				);
				dates.push(
					Object.values(Object.values(fullCsvDataDictArray[i])[0])[j]["Date"]
				);
			}

			var eachCsv = getChartFormattedData(sales, customers, promo, dates);
			console.log("full", eachCsv);
			fullCsvData[Object.keys(fullCsvDataDictArray[i])[0]] = eachCsv;
		}
		setCompleteCsvData(() => {
			return fullCsvData;
		});
	};

	const fetchSlicedCsvData = () => {
		setShowChartStatus(true);
		fetchFullCsvData();

		// Pagination
		const indexOfLastItem = currentPage * itemPerPage;
		const indexOfFirstItem = indexOfLastItem - itemPerPage;

		var slicedCsvData = {};
		var slicedCsvDataDictArray = [];
		for (
			var i = 0;
			i < Object.keys(incomingCsvData["csvDataFrameArray"]).length;
			i++
		) {
			let tempDict = {};
			tempDict[Object.keys(incomingCsvData["csvDataFrameArray"])[i]] =
				Object.values(incomingCsvData["csvDataFrameArray"])[i];
			slicedCsvDataDictArray.push(tempDict);
		}

		const currentItems = slicedCsvDataDictArray.slice(
			indexOfFirstItem,
			indexOfLastItem
		);

		for (var i = 0; i < currentItems.length; i++) {
			var sales = [];
			var customers = [];
			var promo = [];
			var dates = [];

			for (
				var j = 0;
				j < Object.values(Object.values(currentItems[i])[0]).length;
				j++
			) {
				sales.push(
					Object.values(Object.values(currentItems[i])[0])[j]["Sales"]
				);
				customers.push(
					Object.values(Object.values(currentItems[i])[0])[j]["Customers"]
				);
				promo.push(
					Object.values(Object.values(currentItems[i])[0])[j]["Promo"]
				);
				dates.push(Object.values(Object.values(currentItems[i])[0])[j]["Date"]);
			}

			var eachCsv = getChartFormattedData(sales, customers, promo, dates);
			console.log("sliced", eachCsv);
			slicedCsvData[Object.keys(currentItems[i])[0]] = eachCsv;
		}

		setCsvData(() => {
			return slicedCsvData;
		});
	};

	const getSearchTerm = () => {
		setSearchTerm(inputElement.current.value);
		if (inputElement.current.value.length != 0) {
			const newData = Object.keys(completeCsvData).filter((companyName) => {
				// console.log(inputElement.current.value);
				return Object.values(companyName)
					.join("")
					.toLowerCase()
					.includes(inputElement.current.value.toLowerCase());
			});
			var temp = {};
			for (var index = 0; index < newData.length; index++) {
				temp[newData[index]] = completeCsvData[newData[index]];
			}
			setSearchResult(temp);
		}
	};

	function getChartFormattedData(sales, customers, promo, dates) {
		return {
			labels: dates,
			datasets: [
				{
					label: "Sales Trend",
					data: sales,
					backgroundColor: "rgba(255, 99, 132, 0.2)",
					borderColor: "#FBFF00",
					// borderColor: Utils.CHART_COLORS.blue,
					borderWidth: 2,
				},
				{
					label: "Promotion Trend",
					data: promo,
					backgroundColor: "rgba(255, 99, 132, 0.2)",
					borderColor: "#49FF00",
					// borderColor: Utils.CHART_COLORS.blue,
					borderWidth: 2,
				},
				{
					label: "Customers",
					data: customers,
					backgroundColor: "rgba(255, 99, 132, 0.2)",
					borderColor: "#FF9300",
					// borderColor: Utils.CHART_COLORS.blue,
					borderWidth: 2,
				},
			],
		};
	}
	const handleClick = (e) => {
		setCurrentPage(() => {
			return Number(e.target.id);
		});
		fetchSlicedCsvData();
	};

	return (
		<>
			<div className="ui-search">
				<input
					type="text"
					placeholder="Search by company name..."
					name="searchTerm"
					id=""
					className="prompt"
					onChange={getSearchTerm}
					ref={inputElement} // binding use ref to input tag
				/>
			</div>
			<button
				onClick={fetchSlicedCsvData}
				className="btn btn-sm animated-button thar-one"
				style={{ width: "10%", fontSize: "15px" }}
			>
				Show Charts
			</button>
			{showChartStatus && (
				<>
					<div className="graphRow">
						{searchTerm.length == 0
							? Object.keys(csvData).map((singleCsv) => {
									return (
										<div className="graph" key={singleCsv}>
											<p>{singleCsv}</p>
											<Line data={csvData[singleCsv]} />
										</div>
									);
							  })
							: Object.keys(searchResult).map((singleCsv) => {
									return (
										<div className="graph" key={singleCsv}>
											<p>{singleCsv}</p>
											<Line data={completeCsvData[singleCsv]} />
										</div>
									);
							  })}
					</div>

					{pages.map((number) => {
						return (
							<div
								className="btnDiv"
								style={{ marginLeft: "30%", marginTop: "30px" }}
							>
								<button
									key={number}
									id={number}
									onClick={handleClick}
									className="printAndSend"
								>
									{number}
								</button>
							</div>
						);
					})}
				</>
			)}
		</>
	);
};

export default Report;
