#ifndef DATA_H_
#define DATA_H_

#include <vector>
#include <map>

class DataMatrix {

public:
	DataMatrix();

	struct FeaturePoint {
		int sample_index;
		double value;
	};

	struct SamplePoint {
		std::map<int, double> features;
	};

	int Load(const char *filename);
	void SetTargets(const std::vector<double>& targets);
	const SamplePoint& GetRow(unsigned int index) const;
	const std::vector<FeaturePoint>& GetColumn(unsigned int index) const;
	const std::vector<double>& GetTargets() const;

	unsigned int Size();
	unsigned int Dimension();

private:

	std::map<int, std::vector<FeaturePoint> > column_data_;
	std::vector<SamplePoint> row_data_;
	std::vector<double> targets_;

};

#endif
