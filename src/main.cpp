#include <vtkAutoInit.h>

VTK_MODULE_INIT(vtkRenderingOpenGL2)
VTK_MODULE_INIT(vtkInteractionStyle)

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#include <iomanip> // setprecision
#include <sstream> // stringstream
#include <string>

#include <cpd/nonrigid.hpp>

#include <iostream>
#include <fstream>

#include <vtkPLYReader.h>
#include <vtkOBJReader.h>

#include <vtkArrowSource.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkProperty.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkMath.h>

#include <vtkVersion.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVertexGlyphFilter.h>

#include <vtkQuadricDecimation.h>

#include <vtkObjectFactory.h>
#include <vtkInteractorStyleTrackballCamera.h>

#include <vtkTextActor.h>
#include <vtkTextProperty.h>

#include <vtkTextWidget.h>
#include <vtkTextRepresentation.h>
#include <vtkCoordinate.h>
#include <vtkCommand.h>

#include <vtkDecimatePro.h>

#include <vtkPoints.h>

#include <vtkLookupTable.h>
#include <vtkDoubleArray.h>
#include <vtkCellData.h>
#include <vtkPolyData.h>

#include <vtkVersion.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkPolyData.h>
#include <vtkCubeSource.h>
#include <vtkSphereSource.h>
#include <vtkLegendBoxActor.h>
#include <vtkNamedColors.h>

#include <vtkSmartPointer.h>

#include <vtkAxesActor.h>
#include <vtkSimplePointsReader.h>

#include "gplvm.hpp"

int meshCount = 3;
int currentMesh = 0;
const int templateNum = 2;
const int startNum = 1;

bool dense = false;

int numParams = 2;

const double points = 35.0;

const double step = 0.05;
double interpolation = step;

const double param_step = 0.1;
std::vector<double> paramStepMod;

const int R = 5; // Number of arrows in the field
const double size = 10.0; // Size of arrows relative to density

vtkSmartPointer<vtkPolyData> templateMesh;
vtkSmartPointer<vtkPolyData> deformedMesh;
std::vector<vtkSmartPointer< vtkPolyData > > observedMeshes;
vtkSmartPointer<vtkPolyData> templatePoints;
std::vector< vtkSmartPointer< vtkPolyData > > observedPointsVector;
std::vector<vtkSmartPointer< vtkActor > > transformArrows;
std::vector<vtkSmartPointer< vtkActor > > fieldArrows;
std::vector<vtkSmartPointer< vtkArrowSource > > transformArrowsSource;
std::vector<vtkSmartPointer< vtkArrowSource > > fieldArrowsSource;
std::vector<vtkSmartPointer< vtkTransformPolyDataFilter > > transformArrowsTransforms;
std::vector<vtkSmartPointer< vtkTransformPolyDataFilter > > fieldArrowsTransforms;
std::vector<vtkSmartPointer< vtkLookupTable > > luts;

vtkSmartPointer<vtkAxesActor> templateAxes;
vtkSmartPointer<vtkAxesActor> deformedAxes;
std::vector< vtkSmartPointer< vtkAxesActor > > observedAxes;
vtkSmartPointer<vtkTransform> templateAxesTransform;
double templateAxesPosition[3];
const double axesScale = 0.25;

arma::mat templateAxesPoints;

double red[3] = {1.0, 0.0, 0.0};
double green[3] = {0.0, 1.0, 0.0};
double blue[3] = {0.0, 0.0, 1.0};
double white[3] = {1.0, 1.0, 1.0};
double lgrey[3] = {0.75, 0.75, 0.75};
double grey[3] = {0.5, 0.5, 0.5};

const int baseText = 0;
const int deformedText = 1;
const int observedText = 2;
const int fieldText = 3;
const int arrowText = 4;
//const int reverseText = 5;
//const int forwardText = 6
const int valueText = 7;
const int paramText = 8;
const int paramValueText = 9;
const int increaseText = 10;
const int decreateText = 11;
//const int modelText = 12;
const int modelValueText = 13;
const int modelTypeText = 14;
const int XSetText = 15;
//const int pointSetText = 16;

arma::mat _T; // Template
arma::mat _Tm; // Template mesh
arma::mat _G; // Kernel (w.r.t. model)
arma::mat _W; // Weights
arma::mat _Wm; // Mean weights
arma::mat _V; // Latent weights
arma::mat _X; // Latent variables
arma::mat _GW; // G * W
arma::mat _Ga; // For the deformed axes
// D = G*W; // Deformed
// W = V*X;
int _M;
int _N;

int currentX = 0;
std::vector<arma::mat> _XsL;
std::vector<arma::mat> _XsNL;

vtkSmartPointer<vtkRenderer> renderer;
vtkSmartPointer<vtkRenderWindow> renderWindow;
vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor;

vtkSmartPointer<vtkPolyDataMapper> baseMapper;
vtkSmartPointer<vtkPolyDataMapper> deformedMapper;
vtkSmartPointer<vtkPolyDataMapper> dataMapper;

vtkSmartPointer<vtkActor> baseActor;
vtkSmartPointer<vtkActor> deformedActor;
vtkSmartPointer<vtkActor> dataActor;
vtkSmartPointer<vtkActor> basePointsActor;
vtkSmartPointer<vtkActor> dataPointsActor;

vtkSmartPointer<vtkLegendBoxActor> legend;

bool r = true;
bool g = true;
bool b = false;
bool a = false;
bool v = true;
bool s = false;

bool linear = false;
bool displayingMesh = true;

int currentParam;

const double beta_k = -2 * std::pow(1, 2); // beta = 1

GPLVM::Ptr gplvm;

arma::mat ComputeG(arma::mat base, arma::mat obs)
{
	const arma::uword M = base.n_rows;
	const arma::uword N = obs.n_rows;

	arma::mat Gr(N, M);

	for (arma::uword i = 0; i < M; ++i)
	{
		arma::mat Tz = arma::repmat(base.row(i), N, 1);
		Gr.col(i) = arma::exp(arma::sum(arma::pow(obs - Tz, 2), 1) / beta_k);
	}

	return Gr;
}

double transformArrow(vtkSmartPointer<vtkTransformPolyDataFilter>& transformPD,
					  double* startPoint, double* endPoint, double* nV = 0, bool field = false)
{
	// Compute a basis
	double normalizedX[3];
	double normalizedY[3];
	double normalizedZ[3];

	// The X axis is a vector from start to end
	vtkMath::Subtract(endPoint, startPoint, normalizedX);
	double length = vtkMath::Norm(normalizedX);
	vtkMath::Normalize(normalizedX);

	// The Z axis is an arbitrary vector cross X
	double arbitrary[3];
	arbitrary[0] = 0;
	arbitrary[1] = 0;
	arbitrary[2] = 1;
	vtkMath::Cross(normalizedX, arbitrary, normalizedZ);
	vtkMath::Normalize(normalizedZ);

	// The Y axis is Z cross X
	vtkMath::Cross(normalizedZ, normalizedX, normalizedY);
	vtkSmartPointer<vtkMatrix4x4> matrix =
			vtkSmartPointer<vtkMatrix4x4>::New();

	// Create the direction cosine matrix
	matrix->Identity();
	for (unsigned int i = 0; i < 3; i++)
	{
		matrix->SetElement(i, 0, normalizedX[i]);
		matrix->SetElement(i, 1, normalizedY[i]);
		matrix->SetElement(i, 2, normalizedZ[i]);
	}

	// Apply the transforms
	vtkSmartPointer<vtkTransform> transform =
			vtkSmartPointer<vtkTransform>::New();
	transform->Translate(startPoint);
	transform->Concatenate(matrix);
	if (field)
	{
		transform->Scale(0.1*(size/R), 0.1*(size/R), 0.1*(size/R));
	}
	else
	{
        transform->Scale(length, length, length);
	}
	transformPD->SetTransform(transform);

	if (nV != 0)
	{
		nV[0] = normalizedX[0];
		nV[1] = normalizedX[1];
		nV[2] = normalizedX[2];
	}
	else { }

	return length;
}

void colorArrow(vtkSmartPointer<vtkLookupTable> lut, double* dist)
{
	// Fill in a few known colors, the rest will be generated if needed
	double shaft[] = {0.5, 0.5, 0.5};
	double tip[] = {std::min(1.0, std::abs(dist[0])),
					std::min(1.0, std::abs(dist[1])),
					std::min(1.0, std::abs(dist[2]))};

	lut->SetTableValue(0, shaft[0], shaft[0], shaft[0], 1);
	lut->SetTableValue(1, shaft[0], shaft[0], shaft[0], 1);
	lut->SetTableValue(2, shaft[0], shaft[0], shaft[0], 1);
	lut->SetTableValue(3, shaft[0], shaft[0], shaft[0], 1);
	lut->SetTableValue(4, shaft[0], shaft[0], shaft[0], 1);
	lut->SetTableValue(5, shaft[0], shaft[0], shaft[0], 1);

	lut->SetTableValue(6, tip[0], tip[0], tip[0], 1);
	lut->SetTableValue(7, tip[0], tip[0], tip[0], 1);
	lut->SetTableValue(8, tip[0], tip[1], tip[2], 1);
	lut->SetTableValue(9, tip[0], tip[1], tip[2], 1);
	lut->SetTableValue(10, tip[0], tip[1], tip[2], 1);
	lut->SetTableValue(11, tip[0], tip[1], tip[2], 1);
	lut->SetTableValue(12, tip[0], tip[1], tip[2], 1);
	lut->SetTableValue(13, tip[0], tip[1], tip[2], 1);
	lut->SetTableValue(14, tip[0], tip[1], tip[2], 1);

	lut->Modified();
}

void updateArrows()
{
	double startPoint[3];
	double endPoint[3];

    //  Update the vector field
    const float start = -1.0;
    const float step = 2.0/(R-1.0);
    arma::mat Z(R * R * R, 3);
    int index = 0;
    for (int i = 0; i < R; i++)
    {
        for (int j = 0; j < R; j++)
        {
            for (int k = 0; k < R; k++)
            {
                Z(index, 0) = i * step + start;
                Z(index, 1) = j * step + start;
                Z(index, 2) = k * step + start;
                index++;
            }
        }
    }
	arma::mat Gz = ComputeG(_T, Z);
	arma::mat GZ = Gz * _W;

    double R3 = R * R * R;
	for (int i = 0; i < R3; i++)
	{
		startPoint[0] = Z(i, 0);
		startPoint[1] = Z(i, 1);
		startPoint[2] = Z(i, 2);

		endPoint[0] = startPoint[0] + GZ(i, 0);
		endPoint[1] = startPoint[1] + GZ(i, 1);
		endPoint[2] = startPoint[2] + GZ(i, 2);

		double nV[3];
        transformArrow(fieldArrowsTransforms[i], startPoint, endPoint, nV, true);
		colorArrow(luts[i], nV);
		fieldArrows[i]->Modified();
	}

    double jump = std::max(1.0, ((double)_Tm.n_rows)/R3);
	int arrowIndex = 0;
    for (double i = 0.0; i < _Tm.n_rows; i+=jump)
	{
        startPoint[0] = _Tm((int)i, 0);
        startPoint[1] = _Tm((int)i, 1);
        startPoint[2] = _Tm((int)i, 2);

        endPoint[0] = startPoint[0] + _GW((int)i, 0);
		endPoint[1] = startPoint[1] + _GW((int)i, 1);
		endPoint[2] = startPoint[2] + _GW((int)i, 2);

        transformArrow(transformArrowsTransforms[arrowIndex], startPoint, endPoint);
		transformArrows[arrowIndex]->Modified();
		arrowIndex++;
	}
}

vtkSmartPointer<vtkActor> makeArrow(double* startPoint, double* endPoint, bool field = false)
{
    // Create an arrow
	vtkSmartPointer<vtkArrowSource> arrowSource =
			vtkSmartPointer<vtkArrowSource>::New();

	// Compute a basis
	double normalizedX[3];
	double normalizedY[3];
	double normalizedZ[3];

	// The X axis is a vector from start to end
	vtkMath::Subtract(endPoint, startPoint, normalizedX);
	double length = vtkMath::Norm(normalizedX);
	vtkMath::Normalize(normalizedX);

	// The Z axis is an arbitrary vector cross X
	double arbitrary[3];
	arbitrary[0] = 0;
	arbitrary[1] = 0;
	arbitrary[2] = 1;
	vtkMath::Cross(normalizedX, arbitrary, normalizedZ);
	vtkMath::Normalize(normalizedZ);

	// The Y axis is Z cross X
	vtkMath::Cross(normalizedZ, normalizedX, normalizedY);
	vtkSmartPointer<vtkMatrix4x4> matrix =
			vtkSmartPointer<vtkMatrix4x4>::New();

	// Create the direction cosine matrix
	matrix->Identity();
	for (unsigned int i = 0; i < 3; i++)
	{
		matrix->SetElement(i, 0, normalizedX[i]);
		matrix->SetElement(i, 1, normalizedY[i]);
		matrix->SetElement(i, 2, normalizedZ[i]);
	}

	// Apply the transforms
	vtkSmartPointer<vtkTransform> transform =
			vtkSmartPointer<vtkTransform>::New();
	transform->Translate(startPoint);
	transform->Concatenate(matrix);

	vtkSmartPointer<vtkLookupTable> lut =
			vtkSmartPointer<vtkLookupTable>::New();
	int indices;

	if (field)
	{
		arrowSource->SetShaftRadius(0.0125);
		arrowSource->SetTipRadius(0.025);
		arrowSource->SetTipLength(0.3);

		// Create cell data
		arrowSource->Update();
		indices = arrowSource->GetOutput()->GetNumberOfCells();
		vtkSmartPointer<vtkDoubleArray> cellData =
				vtkSmartPointer<vtkDoubleArray>::New();
		for (int i = 0; i < indices; i++)
		{
			cellData->InsertNextValue(i);
		}

		// Create a lookup table to map cell data to colors
		lut->SetNumberOfTableValues(indices);
		lut->Build();

		colorArrow(lut, normalizedX);
		luts.push_back(lut);

		arrowSource->Update(); // Force an update so we can set cell data
		arrowSource->GetOutput()->GetCellData()->SetScalars(cellData);

        transform->Scale(0.1*(size/R), 0.1*(size/R), 0.1*(size/R));
	}
	else // Not field
	{
		arrowSource->SetShaftRadius(0.1);
		arrowSource->SetTipRadius(0.25);
        arrowSource->SetTipLength(0.5);
        transform->Scale(length, length, length);
	}

	// Transform the polydata
	vtkSmartPointer<vtkTransformPolyDataFilter> transformPD =
			vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	transformPD->SetTransform(transform);
	transformPD->SetInputConnection(arrowSource->GetOutputPort());

    // Create a mapper and actor for the arrow
	vtkSmartPointer<vtkPolyDataMapper> mapper =
			vtkSmartPointer<vtkPolyDataMapper>::New();
	vtkSmartPointer<vtkActor> actor =
			vtkSmartPointer<vtkActor>::New();
	mapper->SetInputConnection(transformPD->GetOutputPort());

	if (field)
	{
		mapper->SetScalarRange(0, indices);
		mapper->SetLookupTable(lut);
		fieldArrowsSource.push_back(arrowSource);
		fieldArrowsTransforms.push_back(transformPD);
	}
	else
	{
		transformArrowsSource.push_back(arrowSource);
		transformArrowsTransforms.push_back(transformPD);
	}

	actor->SetMapper(mapper);

	return actor;
}

void updateAxes()
{
    arma::mat Ga = ComputeG(_T, templateAxesPoints);
    arma::mat Za = Ga * _W;

    arma::mat deformedAxesPoints = templateAxesPoints + Za;

    arma::mat B(3, 3);
    B.zeros();
    for (int i = 0; i < deformedAxesPoints.n_rows; i++)
    {
        arma::mat w(3, 1); // Base
        w(0, 0) = templateAxesPoints(i%4, 0)-templateAxesPoints(0, 0);
        w(1, 0) = templateAxesPoints(i%4, 1)-templateAxesPoints(0, 1);
        w(2, 0) = templateAxesPoints(i%4, 2)-templateAxesPoints(0, 2);
        w = arma::normalise(w, 2, 0);
        arma::mat v(1, 3); // Deformed
        v(0, 0) = deformedAxesPoints(i, 0)-deformedAxesPoints(0, 0);
        v(0, 1) = deformedAxesPoints(i, 1)-deformedAxesPoints(0, 1);
        v(0, 2) = deformedAxesPoints(i, 2)-deformedAxesPoints(0, 2);
        v = arma::normalise(v, 2, 1);
        B += w*v;
    }
    arma::mat U, V;
    arma::vec s;
    arma::svd(U, s, V, B);
    arma::mat M = arma::eye(3, 3);
    M(2, 2) = arma::det(U) * arma::det(V);
    arma::mat Rot = U * M * V.t();

    double thetaX = std::atan2(Rot(2,1), Rot(2,2)) * interpolation;
    double thetaY = std::atan2(-(2,0), std::sqrt(Rot(2,1)*Rot(2,1)+Rot(2,2)*Rot(2,2))) * interpolation;
    double thetaZ = std::atan2(Rot(1,0), Rot(0,0)) * interpolation;

    arma::mat X(3, 3);
    X(0, 0) = 1; X(0, 1) = 0; X(0, 2) = 0;
    X(1, 0) = 0; X(1, 1) = std::cos(thetaX); X(1, 2) = -std::sin(thetaX);
    X(2, 0) = 0; X(2, 1) = std::sin(thetaX); X(2, 2) = std::cos(thetaX);
    arma::mat Y(3, 3);
    Y(0, 0) = std::cos(thetaY); Y(0, 1) = 0; Y(0, 2) = std::sin(thetaY);
    Y(1, 0) = 0; Y(1, 1) = 1; Y(1, 2) = 0;
    Y(2, 0) = -std::sin(thetaY); Y(2, 1) = 0; Y(2, 2) = std::cos(thetaY);
    arma::mat Z(3, 3);
    Z(0, 0) = std::cos(thetaZ); Z(0, 1) = -std::sin(thetaZ); Z(0, 2) = 0;
    Z(1, 0) = std::sin(thetaZ); Z(1, 1) = std::cos(thetaZ); Z(1, 2) = 0;
    Z(2, 0) = 0; Z(2, 1) = 0; Z(2, 2) = 1;
    Rot = Z*Y*X;
    double t1 = (deformedAxesPoints(0, 0)-templateAxesPoints(0, 0))*interpolation;
    double t2 = (deformedAxesPoints(0, 1)-templateAxesPoints(0, 1))*interpolation;
    double t3 = (deformedAxesPoints(0, 2)-templateAxesPoints(0, 2))*interpolation;
    vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
    matrix->SetElement(0, 0, Rot(0, 0)); matrix->SetElement(0, 1, Rot(0, 1)); matrix->SetElement(0, 2, Rot(0, 2)); matrix->SetElement(0, 3,  t1);
    matrix->SetElement(1, 0, Rot(1, 0)); matrix->SetElement(1, 1, Rot(1, 1)); matrix->SetElement(1, 2, Rot(1, 2)); matrix->SetElement(1, 3,  t2);
    matrix->SetElement(2, 0, Rot(2, 0)); matrix->SetElement(2, 1, Rot(2, 1)); matrix->SetElement(2, 2, Rot(2, 2)); matrix->SetElement(2, 3,  t3);
    matrix->SetElement(3, 0,       0.0); matrix->SetElement(3, 1,       0.0); matrix->SetElement(3, 2,       0.0); matrix->SetElement(3, 3, 1.0);

    vtkSmartPointer<vtkTransform> defTransform =
            vtkSmartPointer<vtkTransform>::New();
    defTransform->PostMultiply();
    defTransform->Concatenate(templateAxesTransform);
    defTransform->Translate(-templateAxesPoints(0, 0), -templateAxesPoints(0, 1), -templateAxesPoints(0, 2));
    defTransform->Concatenate(matrix);
    defTransform->Translate(templateAxesPoints(0, 0), templateAxesPoints(0, 1), templateAxesPoints(0, 2));
    deformedAxes->SetUserTransform(defTransform);
    deformedAxes->Modified();
}

// Define interaction style
class KeyPressInteractorStyle : public vtkInteractorStyleTrackballCamera
{
public:
	static KeyPressInteractorStyle* New();
	vtkTypeMacro(KeyPressInteractorStyle, vtkInteractorStyleTrackballCamera)

	virtual void OnKeyPress()
	{
		// Get the keypress
		vtkRenderWindowInteractor *rwi = this->Interactor;
		std::string key = rwi->GetKeySym();

		double prevInterpolation = interpolation;
		bool paramChanged = false;

		bool changed = false;

		bool arrowsNeedUpdate = false;

		if (key == "r")
		{
			r = !r;
			if (r)
			{
				if (displayingMesh)
				{
					renderer->AddActor(baseActor);
				}
				else
				{
					renderer->AddActor(basePointsActor);
				}
                if (s) renderer->AddActor(templateAxes);
				legend->SetEntryColor(baseText, red);
			}
			else
			{
				if (displayingMesh)
				{
					renderer->RemoveActor(baseActor);
				}
				else
				{
					renderer->RemoveActor(basePointsActor);
				}
                if (s) renderer->RemoveActor(templateAxes);
				legend->SetEntryColor(baseText, grey);
			}
			changed = true;
		}
		else if (key == "g")
		{
			g = !g;
			if (g)
			{
				renderer->AddActor(deformedActor);
                if (s) renderer->AddActor(deformedAxes);
				legend->SetEntryColor(deformedText, green);
			}
			else
			{
				renderer->RemoveActor(deformedActor);
                if (s) renderer->RemoveActor(deformedAxes);
				legend->SetEntryColor(deformedText, grey);
			}
			changed = true;
		}
		else if (key == "b")
		{
			b = !b;
			if (b)
			{
				if (displayingMesh)
				{
					renderer->AddActor(dataActor);
				}
				else
				{
					renderer->AddActor(dataPointsActor);
				}
                if (s) renderer->AddActor(observedAxes[currentMesh]);
				legend->SetEntryColor(observedText, blue);
            }
			else
			{
				if (displayingMesh)
				{
					renderer->RemoveActor(dataActor);
				}
				else
				{
					renderer->RemoveActor(dataPointsActor);
				}
                if (s) renderer->RemoveActor(observedAxes[currentMesh]);
				legend->SetEntryColor(observedText, grey);
			}
			changed = true;
		}
		else if (key == "a")
		{
			a = !a;
			if (a)
			{
				for (int i = 0; i < transformArrows.size(); i++)
				{
					renderer->AddActor(transformArrows[i]);
				}
				legend->SetEntryColor(arrowText, lgrey);
			}
			else
			{
				for (int i = 0; i < transformArrows.size(); i++)
				{
					renderer->RemoveActor(transformArrows[i]);
				}
				legend->SetEntryColor(arrowText, grey);
			}
			changed = true;
		}
		else if (key == "v")
		{
			v = !v;
			if (v)
			{
				for (int i = 0; i < fieldArrows.size(); i++)
				{
					renderer->AddActor(fieldArrows[i]);
				}
				legend->SetEntryColor(fieldText, white);
			}
			else
			{
				for (int i = 0; i < fieldArrows.size(); i++)
				{
					renderer->RemoveActor(fieldArrows[i]);
				}
				legend->SetEntryColor(fieldText, grey);
			}
			changed = true;
		}
		else if (key == "o")
		{
            int previousMesh = currentMesh;
			currentMesh = (currentMesh + 1) % meshCount;
			if (displayingMesh)
			{
				dataMapper->SetInputData(observedMeshes[currentMesh]);
			}
			else
			{
				dataMapper->SetInputData(observedPointsVector[currentMesh]);
			}
			dataMapper->Modified();
            if (s)
            {
                renderer->RemoveActor(observedAxes[previousMesh]);
                renderer->AddActor(observedAxes[currentMesh]);
            }
            else { }
			char valueValue[18];
			std::sprintf(valueValue, "Observed Model: %02d", currentMesh+1);
			legend->SetEntryString(modelValueText, valueValue);
			legend->Modified();
			changed = true;
		}
		else if (key == "Left")
		{
			interpolation = std::max(0.01, interpolation - step);
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << interpolation;
			std::string s = stream.str();
			std::string valueValue = "Transform Value: +" + s;
			legend->SetEntryString(valueText, valueValue.c_str());
			legend->Modified();
			changed = true;
		}
		else if (key == "Right")
		{
            interpolation = std::min(0.99, interpolation + step);
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << interpolation;
			std::string s = stream.str();
			std::string valueValue = "Transform Value: +" + s;
			legend->SetEntryString(valueText, valueValue.c_str());
			legend->Modified();
			changed = true;
		}
		else if (key == "1" || key == "2" || key == "3" || key == "4" || key == "5" || key == "6" || key == "7" || key == "8" || key == "9")
		{
			int num = std::stoi(key);
			if (num <= numParams)
			{
				currentParam = num-1;
				std::string value = "# - Select Parameter: " + key;
				legend->SetEntryString(paramText, value.c_str());
				std::stringstream stream;
				stream << std::fixed << std::setprecision(2) << _X(0, currentParam);
				std::string s = stream.str();
				std::string valueValue;
				if (_X(0, currentParam) >= 0)
				{
					valueValue = "Parameter Value: +" + s;
				}
				else
				{
					valueValue = "Parameter Value:  " + s;
				}
				legend->SetEntryString(paramValueText, valueValue.c_str());
				legend->Modified();
				changed = true;
			}
			else { }
		}
		/*else if (key == "0")
		{
			if (10 <= numParams)
			{
				currentParam = 9;
				legend->SetEntryString(paramText, "# - Select Parameter: 10");
				std::stringstream stream;
				stream << std::fixed << std::setprecision(2) << _X(0, currentParam);
				std::string s = stream.str();
				std::string valueValue;
				if (_X(0, currentParam) >= 0)
				{
					valueValue = "Parameter Value: +" + s;
				}
				else
				{
					valueValue = "Parameter Value:  " + s;
				}
				legend->SetEntryString(paramValueText, valueValue.c_str());
				legend->Modified();
				changed = true;
			}
			else { }
		}*/
		else if (key == "Up")
		{
			_X(0, currentParam) += (param_step * paramStepMod[currentParam]);
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << _X(0, currentParam);
			std::string s = stream.str();
			std::string valueValue;
			if (_X(0, currentParam) >= 0)
			{
				valueValue = "Parameter Value: +" + s;
			}
			else
			{
				valueValue = "Parameter Value:  " + s;
			}
			legend->SetEntryString(paramValueText, valueValue.c_str());
            currentX = 0;
            legend->SetEntryString(XSetText, "x - X: Custom");
			legend->Modified();
			paramChanged = true;
			changed = true;
		}
		else if (key == "Down")
		{
			_X(0, currentParam) -= (param_step * paramStepMod[currentParam]);
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << _X(0, currentParam);
			std::string s = stream.str();
			std::string valueValue;
			if (_X(0, currentParam) >= 0)
			{
				valueValue = "Parameter Value: +" + s;
			}
			else
			{
				valueValue = "Parameter Value:  " + s;
			}
			legend->SetEntryString(paramValueText, valueValue.c_str());
            currentX = 0;
            legend->SetEntryString(XSetText, "x - X: Custom");
			legend->Modified();
			paramChanged = true;
			changed = true;
		}
        else if (key == "l")
        {
            linear = !linear;
            if (linear)
            {
                legend->SetEntryString(modelTypeText, "l - Model: Linear");
                if (currentX > 0)
                {
                    _X = _XsL[currentX-1];
                }
                else { }
            }
            else
            {
                legend->SetEntryString(modelTypeText, "l - Model: Non-Linear");
                if (currentX > 0)
                {
                    _X = _XsNL[currentX-1];
                }
                else { }
            }
            std::stringstream stream;
            stream << std::fixed << std::setprecision(2) << _X(0, currentParam);
            std::string s = stream.str();
            std::string valueValue;
            if (_X(0, currentParam) >= 0)
            {
                valueValue = "Parameter Value: +" + s;
            }
            else
            {
                valueValue = "Parameter Value:  " + s;
            }
            legend->SetEntryString(paramValueText, valueValue.c_str());
            legend->Modified();
            paramChanged = true;
            changed = true;
        }
        else if (key == "x")
        {
            currentX = (currentX + 1) % (_N+1);
            if (currentX == 0)
            {
				_X.zeros();
				legend->SetEntryString(XSetText, "x - X: Mean");
            }
            else
            {
                char XValue[9];
                std::sprintf(XValue, "x - X: %02d", currentX);
                legend->SetEntryString(XSetText, XValue);
                if (linear)
                {
                    _X = _XsL[currentX-1];
                }
                else
                {
                    _X = _XsNL[currentX-1];
                }
            }
			std::stringstream stream;
			stream << std::fixed << std::setprecision(2) << _X(0, currentParam);
			std::string s = stream.str();
			std::string valueValue;
			if (_X(0, currentParam) >= 0)
			{
				valueValue = "Parameter Value: +" + s;
			}
			else
			{
				valueValue = "Parameter Value:  " + s;
			}
			legend->SetEntryString(paramValueText, valueValue.c_str());
            legend->Modified();
            paramChanged = true;
            changed = true;
        }
		else if (key == "p")
		{
			displayingMesh = !displayingMesh;
			if (displayingMesh)
			{
				if (r)
				{
					renderer->RemoveActor(basePointsActor);
					renderer->AddActor(baseActor);
				}
				else { }
				if (b)
				{
					renderer->RemoveActor(dataPointsActor);
					renderer->AddActor(dataActor);
				}
				else { }
			}
			else
			{
				if (r)
				{
					renderer->RemoveActor(baseActor);
					renderer->AddActor(basePointsActor);
				}
				else { }
				if (b)
				{
					renderer->RemoveActor(dataActor);
					renderer->AddActor(dataPointsActor);
				}
				else { }
			}
			/*if (displayingMesh)
			{
				baseMapper->SetInputData(templateMesh);
				dataMapper->SetInputData(observedMeshes[currentMesh]);
			}
			else
			{
				baseMapper->SetInputData(templatePoints);
				dataMapper->SetInputData(observedPointsVector[currentMesh]);
			}
			baseMapper->Modified();
			dataMapper->Modified();*/
			changed = true;
		}
        else if (key == "s")
        {
            s = !s;
            if (s)
            {
                if (b) renderer->AddActor(observedAxes[currentMesh]);
                if (g) renderer->AddActor(deformedAxes);
                if (r) renderer->AddActor(templateAxes);
				arrowsNeedUpdate = true;
            }
            else
            {
                if (b) renderer->RemoveActor(observedAxes[currentMesh]);
                if (g) renderer->RemoveActor(deformedAxes);
                if (r) renderer->RemoveActor(templateAxes);
            }
            changed = true;
        }
		else { }

		if (prevInterpolation != interpolation || paramChanged)
		{
			if (linear)
            {
                _W = _X * _V;
                _W = _W + _Wm;
                _W = _W.t();
                _W.reshape(3, _M);
                _W = _W.t();
            }
            else
            {
                arma::mat WC = gplvm->predict(_X);
                int index = 0;
                for (int i = 0; i < _W.n_rows; i++)
                {
                    for (int j = 0; j < _W.n_cols; j++)
                    {
                        _W(i, j) = WC(0, index) + _Wm(0, index);
                        index++;
                    }
                }
            }

			_GW = _G * _W;
			for (int i = 0; i < _GW.n_rows; i++)
			{
				double* dM = deformedMesh->GetPoint(i);
				double* tM = templateMesh->GetPoint(i);
				dM[0] = tM[0] + (interpolation * _GW(i, 0));
				dM[1] = tM[1] + (interpolation * _GW(i, 1));
				dM[2] = tM[2] + (interpolation * _GW(i, 2));
				deformedMesh->GetPoints()->SetPoint(i, dM);
			}

            deformedMesh->Modified();
            updateAxes();
		}
		else { }

		if (paramChanged || arrowsNeedUpdate)
		{
			updateArrows();
		}
		else { }

		if (changed)
		{
			renderWindow->Render();
		}
		else // Forward events
		{
			vtkInteractorStyleTrackballCamera::OnKeyPress();
		}
	}

};
vtkStandardNewMacro(KeyPressInteractorStyle)

void loadMesh(const char* fileName, vtkPolyData* mesh)
{
    printf("Loading " ANSI_COLOR_YELLOW "ply" ANSI_COLOR_RESET " file " ANSI_COLOR_YELLOW "%s" ANSI_COLOR_RESET "... ", fileName); fflush(stdout);
	vtkPLYReader* sceneReader = vtkPLYReader::New();
	sceneReader->SetFileName(fileName);
	sceneReader->Update();
	mesh->ShallowCopy(sceneReader->GetOutput());
	printf("Loaded %d points.\n", ((int)mesh->GetNumberOfPoints())); fflush(stdout);
}

void loadPoints(const char* fileName, vtkPolyData* points)
{
	printf("Loading " ANSI_COLOR_CYAN "obj" ANSI_COLOR_RESET " file " ANSI_COLOR_CYAN "%s" ANSI_COLOR_RESET "... ", fileName); fflush(stdout);
	vtkOBJReader* sceneReader = vtkOBJReader::New();
	sceneReader->SetFileName(fileName);
	sceneReader->Update();
	points->ShallowCopy(sceneReader->GetOutput());
	vtkSmartPointer<vtkCellArray> vertices =
		vtkSmartPointer<vtkCellArray>::New();
	int n = points->GetNumberOfPoints();
	vtkIdType id[1];
	for (int i = 0; i < n; i++)
	{
		id[0] = i;
		vertices->InsertNextCell(1, id);
	}
	points->SetVerts(vertices);
	points->SetPolys(vertices);
	printf("Loaded %d points.\n", ((int)points->GetNumberOfPoints())); fflush(stdout);
}

void loadAxes(const char* fileName, vtkAxesActor* axes, vtkSmartPointer<vtkTransform> t = 0, double* position = 0)
{
    std::ifstream stream;
    stream.open(fileName);

    char line[64];
    stream.getline(line, 64);
    double x = atof(line);
    stream.getline(line, 64);
    double y = atof(line);
    stream.getline(line, 64);
    double z = atof(line);

    stream.getline(line, 64);
    double i = atof(line);
    stream.getline(line, 64);
    double j = atof(line);
    stream.getline(line, 64);
    double k = atof(line);
    stream.getline(line, 64);
    double w = atof(line);
    w = w / (2.0*M_PI) * 360.0;
    double l = std::sqrt(i*i + j*j + k*k);
    i /= l;
    j /= l;
    k /= l;

    vtkSmartPointer<vtkTransform> transform =
        vtkSmartPointer<vtkTransform>::New();
    transform->PostMultiply();
    transform->RotateWXYZ(w, i, j, k);
    transform->Translate(x, y, z);
    axes->SetUserTransform(transform);
    axes->AxisLabelsOff();
    axes->SetTotalLength(axesScale, axesScale, axesScale);
    axes->SetShaftTypeToCylinder();
    if (t != 0) t = transform;
    if (position != 0)
    {
        position[0] = x;
        position[1] = y;
        position[2] = z;
    }
    else { }
}

void printResults(arma::mat& T, arma::mat& W, float beta, std::string filename)
{
	std::ofstream myfile;
	myfile.open(filename);
	for (int i = 0; i < T.n_rows; i++)
	{
		for (int j = 0; j < T.n_cols; j++)
		{
			myfile << T(i, j) << ", ";
		}
		myfile << "\n";
	}
	myfile << "\n";
	for (int i = 0; i < W.n_rows; i++)
	{
		for (int j = 0; j < W.n_cols; j++)
		{
			myfile << W(i, j) << ", ";
		}
		myfile << "\n";
	}
	myfile << beta;
	myfile.close();
}

void printResults(arma::mat& W, std::string filename)
{
	std::ofstream myfile;
	myfile.open(filename);
	for (int i = 0; i < W.n_rows; i++)
	{
		for (int j = 0; j < W.n_cols; j++)
		{
			myfile << W(i, j) << " ";
		}
		myfile << "\n";
	}
	myfile.close();
}

void addLabels()
{
	legend = vtkSmartPointer<vtkLegendBoxActor>::New();
    legend->SetNumberOfEntries(17);

	vtkSmartPointer<vtkCubeSource> legendBox =
			vtkSmartPointer<vtkCubeSource>::New();
	legendBox->Update();
	if (r) legend->SetEntry(0, legendBox->GetOutput(), "r - Template", red);
	else legend->SetEntry(0, legendBox->GetOutput(), "r - Template", grey);
	if (g) legend->SetEntry(1, legendBox->GetOutput(), "g - Deformed", green);
	else legend->SetEntry(1, legendBox->GetOutput(), "g - Deformed", grey);
	if (b) legend->SetEntry(2, legendBox->GetOutput(), "b - Observed", blue);
	else legend->SetEntry(2, legendBox->GetOutput(), "b - Observed", grey);
	if (v) legend->SetEntry(3, legendBox->GetOutput(), "v - Warp Field", white);
	else legend->SetEntry(3, legendBox->GetOutput(), "v - Warp Field", grey);
	if (a) legend->SetEntry(4, legendBox->GetOutput(), "a - Warp", lgrey);
	else legend->SetEntry(4, legendBox->GetOutput(), "a - Warp", grey);
	legend->SetEntry(5, legendBox->GetOutput(), "<- - Reverse Transform", white);
	legend->SetEntry(6, legendBox->GetOutput(), "-> - Forward Transform", white);
	legend->SetEntry(7, legendBox->GetOutput(), "Transform Value: 0.01", white);
	legend->SetEntry(8, legendBox->GetOutput(), "# - Select Parameter: 1", white);
	legend->SetEntry(9, legendBox->GetOutput(), "Parameter Value: +0.00", white);
	legend->SetEntry(10, legendBox->GetOutput(), "^ - Increase Value", white);
	legend->SetEntry(11, legendBox->GetOutput(), "v - Decrease Value", white);
	legend->SetEntry(12, legendBox->GetOutput(), "o - Show Next Observed Model", white);
	legend->SetEntry(13, legendBox->GetOutput(), "Observed Model: 01", white);
	legend->SetEntry(14, legendBox->GetOutput(), "l - Model: Non-Linear", white);
	legend->SetEntry(15, legendBox->GetOutput(), "x - X: Mean", white);
	legend->SetEntry(16, legendBox->GetOutput(), "p - Switch: Mesh <-> Points", white);
    legend->SetEntry(17, legendBox->GetOutput(), "s - Axes", white);

	legend->GetPositionCoordinate()->SetCoordinateSystemToView();
	legend->GetPosition2Coordinate()->SetCoordinateSystemToView();
	//legend->GetPositionCoordinate()->SetValue(-1.0, 0.85);
	//legend->GetPosition2Coordinate()->SetValue(10.0, 1.0);
	// LB -> RT
	legend->GetPositionCoordinate()->SetValue(-0.95, -0.95);
	legend->GetPosition2Coordinate()->SetValue(-0.45, -0.45);

	legend->BoxOff();
	legend->UseBackgroundOff();

	// Add the actors to the scene
	renderer->AddActor(legend);
	renderWindow->Render();
}

void addMeshes()
{
	// Visualization
	baseMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	baseMapper->SetInputData(templateMesh);

	deformedMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	deformedMapper->SetInputData(deformedMesh);

	dataMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	dataMapper->SetInputData(observedMeshes[currentMesh]);

	baseActor = vtkSmartPointer<vtkActor>::New();
	baseActor->GetProperty()->SetColor(red);
	baseActor->SetMapper(baseMapper);
	baseActor->GetProperty()->SetPointSize(15);

	deformedActor = vtkSmartPointer<vtkActor>::New();
	deformedActor->GetProperty()->SetColor(green);
	deformedActor->SetMapper(deformedMapper);
	deformedActor->GetProperty()->SetPointSize(15);

	dataActor = vtkSmartPointer<vtkActor>::New();
	dataActor->GetProperty()->SetColor(blue);
	dataActor->SetMapper(dataMapper);
	dataActor->GetProperty()->SetPointSize(15);

	vtkSmartPointer<vtkPolyDataMapper> basePointsMapper =
			vtkSmartPointer<vtkPolyDataMapper>::New();
	basePointsMapper->SetInputData(templatePoints);

	vtkSmartPointer<vtkPolyDataMapper> dataPointsMapper =
			vtkSmartPointer<vtkPolyDataMapper>::New();
	dataPointsMapper->SetInputData(observedPointsVector[currentMesh]);

	basePointsActor = vtkSmartPointer<vtkActor>::New();
	basePointsActor->GetProperty()->SetColor(red);
	basePointsActor->SetMapper(basePointsMapper);
	basePointsActor->GetProperty()->SetPointSize(15);

	dataPointsActor = vtkSmartPointer<vtkActor>::New();
	dataPointsActor->GetProperty()->SetColor(blue);
	dataPointsActor->SetMapper(dataPointsMapper);
	dataPointsActor->GetProperty()->SetPointSize(15);

	if (displayingMesh)
	{
		if (b) renderer->AddActor(dataActor);
		if (g) renderer->AddActor(deformedActor);
		if (r) renderer->AddActor(baseActor);
	}
	else
	{
		if (b) renderer->AddActor(dataPointsActor);
		if (g) renderer->AddActor(deformedActor);
		if (r) renderer->AddActor(basePointsActor);
	}

    if (s)
    {
        if (b) renderer->AddActor(observedAxes[currentMesh]);
        if (g) renderer->AddActor(deformedAxes);
        if (r) renderer->AddActor(templateAxes);
    }
    else { }
}

void addField()
{
	double startPoint[3];
	double endPoint[3];

    //  Create the vector field
    const float start = -1.0;
    const float step = 2.0/(R-1.0);
	arma::mat Z(R * R * R, 3);
	int index = 0;
	for (int i = 0; i < R; i++)
	{
		for (int j = 0; j < R; j++)
		{
			for (int k = 0; k < R; k++)
			{
				Z(index, 0) = i * step + start;
				Z(index, 1) = j * step + start;
				Z(index, 2) = k * step + start;
				index++;
			}
		}
	}
	arma::mat Gz = ComputeG(_T, Z);
	arma::mat GZ = Gz * _W;

	int R3 = R * R * R;
	for (int i = 0; i < R3; i++)
	{
		startPoint[0] = Z(i, 0);
		startPoint[1] = Z(i, 1);
		startPoint[2] = Z(i, 2);

		endPoint[0] = startPoint[0] + GZ(i, 0);
		endPoint[1] = startPoint[1] + GZ(i, 1);
		endPoint[2] = startPoint[2] + GZ(i, 2);

		vtkSmartPointer<vtkActor> actor = makeArrow(startPoint, endPoint, true);
		fieldArrows.push_back(actor);

        // Add the actor to the scene
		if (v) renderer->AddActor(actor);
	}
}

void displayResults()
{
	if (linear)
    {
        _W = _X * _V;
        _W = _W + _Wm;
        _W = _W.t();
        _W.reshape(3, _M);
        _W = _W.t();
    }
    else
    {
        arma::mat WC = gplvm->predict(_X);
        int index = 0;
        for (int i = 0; i < _W.n_rows; i++)
        {
            for (int j = 0; j < _W.n_cols; j++)
            {
                _W(i, j) = WC(0, index) + _Wm(0, index);
                index++;
            }
        }
	}

	// Create the model transformation
	int q = templateMesh->GetNumberOfPoints();
    _Tm.set_size(q, 3);
	for (int i = 0; i < q; i++)
	{
		double* point = templateMesh->GetPoint(i);
        _Tm(i, 0) = point[0];
        _Tm(i, 1) = point[1];
        _Tm(i, 2) = point[2];
	}
    _G = ComputeG(_T, _Tm);
	_GW = _G * _W;

    arma::mat axes(1, 3);
    axes(0, 0) = templateAxesPosition[0];
    axes(0, 1) = templateAxesPosition[1];
    axes(0, 2) = templateAxesPosition[2];
    _Ga = ComputeG(_T, axes);

	double startPoint[3];
	double endPoint[3];
    double count = R * R * R;
    double jump = std::max(1.0, ((double)_Tm.n_rows)/count);
    for (double i = 0.0; i < _Tm.n_rows; i+=jump)
	{
        startPoint[0] = _Tm((int)i, 0);
        startPoint[1] = _Tm((int)i, 1);
        startPoint[2] = _Tm((int)i, 2);

		endPoint[0] = startPoint[0] + _GW((int)i, 0);
		endPoint[1] = startPoint[1] + _GW((int)i, 1);
		endPoint[2] = startPoint[2] + _GW((int)i, 2);

		vtkSmartPointer<vtkActor> actor = makeArrow(startPoint, endPoint);
		transformArrows.push_back(actor);

        // Add the actor to the scene
		if (a) renderer->AddActor(actor);
	}

	addField();

	addMeshes();

	addLabels();

	// Do this so the modified mesh is visible in the first frame
	for (int i = 0; i < _GW.n_rows; i++)
	{
		double* dM = deformedMesh->GetPoint(i);
		double* tM = templateMesh->GetPoint(i);
		dM[0] = tM[0] + (interpolation * _GW(i, 0));
		dM[1] = tM[1] + (interpolation * _GW(i, 1));
		dM[2] = tM[2] + (interpolation * _GW(i, 2));
		deformedMesh->GetPoints()->SetPoint(i, dM);
    }

	deformedMesh->Modified();
    updateAxes();

    // Render and interact
	renderWindow->Render();
	renderWindowInteractor->Start();
}

void normalize(vtkSmartPointer<vtkPolyData> loadedPoints, arma::mat& matrix,
               vtkSmartPointer<vtkPolyData> loadedMesh = 0, vtkSmartPointer<vtkAxesActor> axes = 0,
               double* position = 0)
{
	// Center and resize the template
	int n = loadedPoints->GetNumberOfPoints();
	matrix.set_size(n, 3);

	double minX, minY, minZ;
	minX = minY = minZ = INT_MAX;
	double maxX, maxY, maxZ;
	maxX = maxY = maxZ = INT_MIN;

	for (int i = 0; i < n; i++)
	{
		matrix(i, 0) = loadedPoints->GetPoint(i)[0];
		matrix(i, 1) = loadedPoints->GetPoint(i)[1];
		matrix(i, 2) = loadedPoints->GetPoint(i)[2];

		minX = std::min(minX, matrix(i, 0));
		minY = std::min(minY, matrix(i, 1));
		minZ = std::min(minZ, matrix(i, 2));

		maxX = std::max(maxX, matrix(i, 0));
		maxY = std::max(maxY, matrix(i, 1));
		maxZ = std::max(maxZ, matrix(i, 2));
	}

	double medX = (maxX + minX) / 2.0;
	double medY = (maxY + minY) / 2.0;
	double medZ = (maxZ + minZ) / 2.0;

	matrix.col(0) -= medX;
	matrix.col(1) -= medY;
	matrix.col(2) -= medZ;

    for (int i = 0; i < n; i++)
    {
        double* point = loadedPoints->GetPoint(i);
        loadedPoints->GetPoints()->SetPoint(i, point[0]-medX, point[1]-medY, point[2]-medZ);
    }

	if (loadedMesh != 0)
	{
		int nn = loadedMesh->GetNumberOfPoints();
		for (int i = 0; i < nn; i++)
		{
			double* point = loadedMesh->GetPoint(i);
			loadedMesh->GetPoints()->SetPoint(i, point[0]-medX, point[1]-medY, point[2]-medZ);
		}
    }
    else { }

    if (axes != 0)
    {
        vtkSmartPointer<vtkTransform> fullTransform =
                vtkSmartPointer<vtkTransform>::New();
        vtkSmartPointer<vtkLinearTransform> transform = axes->GetUserTransform();
        fullTransform->PostMultiply();
        fullTransform->Concatenate(transform);
        fullTransform->Translate(-medX, -medY, -medZ);
        axes->SetUserTransform(fullTransform);
    }
    else { }

    if (position != 0)
    {
        position[0] -= medX;
        position[1] -= medY;
        position[2] -= medZ;
    }
    else { }
}

int main()
{
	templateMesh = vtkSmartPointer<vtkPolyData>::New();
	deformedMesh = vtkSmartPointer<vtkPolyData>::New();

	templatePoints = vtkSmartPointer<vtkPolyData>::New();

    // Create a renderer, render window, and interactor
	renderer = vtkSmartPointer<vtkRenderer>::New();
	renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	renderWindow->SetWindowName("Mug Simulator 2017");
	renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow);

	vtkSmartPointer<KeyPressInteractorStyle> style = vtkSmartPointer<KeyPressInteractorStyle>::New();
	renderWindowInteractor->SetInteractorStyle(style);
	style->SetCurrentRenderer(renderer);

	vtkSmartPointer<vtkPolyData> observedPoints;
	vtkSmartPointer<vtkPolyData> observedMesh;

	printf("\n");

	// Template
    printf("Loading " ANSI_COLOR_RED "template" ANSI_COLOR_RESET "...\n");
	char tempMesh[17];
	std::sprintf(tempMesh, "../data/mug%02d.ply", templateNum);
	loadMesh(tempMesh, templateMesh);
	char temp[19];
	if (dense)
	{
		std::sprintf(temp, "../data/mug%02d_d.obj", templateNum);
	}
	else
	{
		std::sprintf(temp, "../data/mug%02d_r.obj", templateNum);
	}
	loadPoints(temp, templatePoints);

    char tempAxes[20];
    std::sprintf(tempAxes, "../data/mug%02d_g.pose", templateNum);
    templateAxes = vtkSmartPointer<vtkAxesActor>::New();
    templateAxesTransform = vtkSmartPointer<vtkTransform>::New();
    loadAxes(tempAxes, templateAxes, templateAxesTransform, templateAxesPosition);
    templateAxes->GetXAxisShaftProperty()->SetColor(1.0, 0.0, 0.0);
    templateAxes->GetYAxisShaftProperty()->SetColor(1.0, 0.0, 0.0);
    templateAxes->GetZAxisShaftProperty()->SetColor(1.0, 0.0, 0.0);
    templateAxes->Modified();
    deformedAxes = vtkSmartPointer<vtkAxesActor>::New();
    loadAxes(tempAxes, deformedAxes);
    deformedAxes->GetXAxisShaftProperty()->SetColor(0.0, 1.0, 0.0);
    deformedAxes->GetYAxisShaftProperty()->SetColor(0.0, 1.0, 0.0);
    deformedAxes->GetZAxisShaftProperty()->SetColor(0.0, 1.0, 0.0);
    deformedAxes->Modified();

	// Observed
    printf("Loading " ANSI_COLOR_BLUE "observed" ANSI_COLOR_RESET "...\n");
	int end = meshCount+startNum;
	if (templateNum >= startNum) end++;
	for (int i = startNum; i < end; i++)
	{
		if (i != templateNum)
		{
			char obs[19];
			if (dense)
			{
				std::sprintf(obs, "../data/mug%02d_d.obj", i);
			}
			else
			{
				std::sprintf(obs, "../data/mug%02d_r.obj", i);
			}
			observedPoints = vtkSmartPointer<vtkPolyData>::New();
			loadPoints(obs, observedPoints);
			observedPointsVector.push_back(observedPoints);

			char obsMesh[17];
			std::sprintf(obsMesh, "../data/mug%02d.ply", i);
			observedMesh = vtkSmartPointer<vtkPolyData>::New();
			loadMesh(obsMesh, observedMesh);
			observedMeshes.push_back(observedMesh);

            char obsAxes[20];
            std::sprintf(obsAxes, "../data/mug%02d_g.pose", i);
            vtkSmartPointer<vtkAxesActor> observedAxis = vtkSmartPointer<vtkAxesActor>::New();
            loadAxes(obsAxes, observedAxis);
            observedAxis->GetXAxisShaftProperty()->SetColor(0.0, 0.0, 1.0);
            observedAxis->GetYAxisShaftProperty()->SetColor(0.0, 0.0, 1.0);
            observedAxis->GetZAxisShaftProperty()->SetColor(0.0, 0.0, 1.0);
            observedAxis->Modified();
            observedAxes.push_back(observedAxis);
		}
		else { }
	}

	printf("\n");

	observedMesh = observedMeshes[currentMesh];

	arma::mat templateMatrix;
    normalize(templatePoints, templateMatrix, templateMesh, templateAxes, templateAxesPosition);
	deformedMesh->DeepCopy(templateMesh);
    deformedAxes->SetUserTransform(templateAxes->GetUserTransform());
    templateAxesTransform = vtkSmartPointer<vtkTransform>::New();
    templateAxesTransform->Concatenate(templateAxes->GetUserTransform());
	int m = templateMatrix.n_rows;

    vtkSmartPointer<vtkPoints> templateAxesPointArray =
            vtkSmartPointer<vtkPoints>::New();
    templateAxesPointArray->InsertNextPoint(0.0+templateAxesPosition[0], 0.0+templateAxesPosition[1], 0.0+templateAxesPosition[2]);
    templateAxesPointArray->InsertNextPoint(0.1+templateAxesPosition[0], 0.0+templateAxesPosition[1], 0.0+templateAxesPosition[2]);
    templateAxesPointArray->InsertNextPoint(0.0+templateAxesPosition[0], 0.1+templateAxesPosition[1], 0.0+templateAxesPosition[2]);
    templateAxesPointArray->InsertNextPoint(0.0+templateAxesPosition[0], 0.0+templateAxesPosition[1], 0.1+templateAxesPosition[2]);

    templateAxesTransform->TransformPoints(templateAxesPointArray, templateAxesPointArray);

    int pointCount = templateAxesPointArray->GetNumberOfPoints();
    templateAxesPoints = arma::mat(pointCount, 3);
    for (int i = 0; i < pointCount; i++)
    {
        double point[3];
        templateAxesPointArray->GetPoint(i, point);
        templateAxesPoints(i, 0) = point[0];
        templateAxesPoints(i, 1) = point[1];
        templateAxesPoints(i, 2) = point[2];
    }

	_W.set_size(meshCount, 3*m);
    std::vector<arma::mat> Ws;
	int printNum = startNum;
	for (int i = 0; i < meshCount; i++)
	{
		arma::mat observedMatrix;
        normalize(observedPointsVector[i], observedMatrix, observedMeshes[i], observedAxes[i]);
		int n = observedMatrix.n_rows;

        printf("" ANSI_COLOR_MAGENTA "Matching" ANSI_COLOR_RED " template point set (%d points) " ANSI_COLOR_MAGENTA "with"
               ANSI_COLOR_BLUE " observed point set #%d (%d points)" ANSI_COLOR_MAGENTA "..." ANSI_COLOR_RESET "\n",
               m, (i+1), n); fflush(stdout);

		cpd::Nonrigid nonrigid(0.000001, 150, 0.1, false, 1.0, 1.0, 1.0);

        cpd::Registration::ResultPtr result = nonrigid.run(observedMatrix, templateMatrix);

		// W = m x 3
		int index = 0;
		for (int r = 0; r < m; r++)
		{
			for (int c = 0; c < 3; c++)
			{
				_W(i, index) = result->W(r, c);
				index++;
			}
		}

		char obsWResult[11];
		if (i == templateNum) printNum++;
		std::sprintf(obsWResult, "mug%02d_W.txt", printNum++);
        printResults(result->W, obsWResult);

        Ws.push_back(result->W);
	}

	_M = m;
    _N = meshCount;
    _Wm = arma::mean(_W, 0);

	arma::mat Wc(_W.n_rows, _W.n_cols);
	for (int i = 0; i < _W.n_rows; i++)
	{
		Wc.row(i) = _W.row(i) - _Wm.row(0);
	}
    arma::mat score;
    arma::vec ev;
    arma::princomp(_V, score, ev, Wc);
    int rows = _V.n_rows;
    numParams = std::min(rows, numParams);
	_V = _V.head_rows(numParams);

	_X.set_size(1, numParams);
	_X.zeros();

	_T = templateMatrix;

	currentParam = 0;

	std::vector<double> mins;
	mins.resize(numParams);
	std::vector<double> maxs;
	maxs.resize(numParams);
	std::vector<double> avgs;
	for (int i = 0; i < numParams; i++)
	{
		mins.push_back(INT_MAX);
		maxs.push_back(INT_MIN);
		avgs.push_back(0);
	}

	printNum = startNum;
    for (int i = 0; i < meshCount; i++)
	{
		arma::mat Y = Ws[i];
		Y = Y.t();
		Y.reshape(3*m, 1);
		Y = Y.t();
		Y = Y - _Wm;

		char obsXResult[11];
		if (i == templateNum) printNum++;
		std::sprintf(obsXResult, "mug%02d_X.txt", printNum++);
		arma::mat Q = Y * _V.t();
        _XsL.push_back(Q);
		printResults(Q, obsXResult);

		for (int i = 0; i < numParams; i++)
		{
			mins[i] = std::min(mins[i], Q(0, i));
			maxs[i] = std::max(maxs[i], Q(0, i));
			avgs[i] += std::abs(Q(0, i));
		}
    }

	for (int i = 0; i < numParams; i++)
	{
		//paramStepMod.push_back(maxs[i] - mins[i]);
        //paramStepMod.push_back(avgs[i] / numParams);
		paramStepMod.push_back(param_step);
	}

    char vResult[9];
	std::sprintf(vResult, "mug_V.txt");
    printResults(_V, vResult);

    char wResult[9];
    std::sprintf(wResult, "mug_W.txt");
    printResults(_W, wResult);

    printResults(_T, _Wm, 1, "results.txt");

    arma::mat Y(Wc.n_rows, Wc.n_cols);
    for (int i = 0; i < Wc.n_rows; i++)
    {
        for (int j = 0; j < Wc.n_cols; j++)
        {
            Y(i, j) = Wc(i, j);
        }
    }

	printf("\n");
	printf("" ANSI_COLOR_GREEN "Learning...\n" ANSI_COLOR_RESET);

    google::InitGoogleLogging("cpd_vtk");
    Compound_Kernel::Ptr kernel = Compound_Kernel::New();
    kernel->addKernel(RBF_Kernel::New());
    //kernel->addKernel(Linear_Kernel::New());
    gplvm = GPLVM::New(Wc, numParams, kernel);
    gplvm->learn();

	printf("" ANSI_COLOR_GREEN "Learning successful!\n" ANSI_COLOR_RESET);
	fflush(stdout);

    arma::mat CX = gplvm->getX();
    for (int i = 0; i < _N; i++)
    {
        arma::mat neXt(1, numParams);
        for (int j = 0; j < numParams; j++)
        {
            neXt(0, j) = CX(i, j);
        }
        _XsNL.push_back(neXt);
	}

	_W.resize(_M, 3);

	displayResults();

	printf("\n");

	return 0;
}
