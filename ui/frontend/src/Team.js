import { TeamMembers } from "./data";
import Card from "react-bootstrap/Card";
import Image from "react-bootstrap/Image";

const Team = () => {
	return (
		<>
			<h3
				style={{ textAlign: "center", color: "#DADADA", paddingBottom: "30px" }}
			>
				Team
			</h3>
			<main id="team">
				{TeamMembers.map((team) => {
					const { id, name, image } = team;

					return (
						<Card
							key={id}
							className="team-card"
							style={{ paddingLeft: "50px" }}
						>
							<Image src={image} roundedCircle className="team-img" />
							<Card.Body style={{ marginRight: "30px" }}>
								<Card.Title className="text-center">{name}</Card.Title>
							</Card.Body>
						</Card>
					);
				})}
			</main>
		</>
	);
};

export default Team;
