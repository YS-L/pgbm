#ifndef DATA_H_
#define DATA_H_

#include <vector>
#include <map>

class DataMatrix {

public:
	DataMatrix();

	struct FeaturePoint {
		int sample_index;
		int feature_index;
		double value;
	};

	int Load(const char *filename);
	const std::vector<FeaturePoint>& GetRow(int index) const;
	const std::vector<FeaturePoint>& GetColumn(int index) const;
	const std::vector<double>& GetTargets() const;

	unsigned int Size();
	unsigned int Dimension();
private:

	std::map<int, std::vector<FeaturePoint> > column_data_;
	std::vector<std::vector<FeaturePoint> > row_data_;
	std::vector<double> targets_;

};

#endif
